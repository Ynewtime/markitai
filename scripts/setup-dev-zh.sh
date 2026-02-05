#!/bin/sh
# Markitai 环境配置脚本 (开发者版)
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

    # 远程执行 - 开发者版需要本地克隆
    echo ""
    echo "================================================"
    echo "  开发者版需要本地仓库"
    echo "================================================"
    echo ""
    echo "  请先克隆仓库:"
    echo ""
    echo "    git clone https://github.com/Ynewtime/markitai.git"
    echo "    cd markitai"
    echo "    ./scripts/setup-dev-zh.sh"
    echo ""
    echo "  或使用用户版快速安装:"
    echo ""
    echo "    curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup-zh.sh | sh"
    echo ""
    exit 1
}

load_library

# ============================================================
# 开发者版专用函数
# ============================================================

# 获取项目根目录（脚本目录的父目录）
get_project_root() {
    dirname "$SCRIPT_DIR"
}

# 中文欢迎信息（开发者版）
zh_print_welcome_dev() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}Markitai 开发环境配置${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  本脚本将配置:\n"
    printf "    ${GREEN}•${NC} Python 虚拟环境及所有依赖\n"
    printf "    ${GREEN}•${NC} pre-commit hooks 代码质量检查\n"
    printf "\n"
    printf "  可选组件:\n"
    printf "    ${YELLOW}•${NC} Playwright - 浏览器自动化\n"
    printf "    ${YELLOW}•${NC} LLM CLI 工具 - Claude Code / Copilot\n"
    printf "    ${YELLOW}•${NC} LLM Python SDKs - 程序化 LLM 访问\n"
    printf "\n"
    printf "  ${BOLD}随时按 Ctrl+C 取消${NC}\n"
    printf "\n"
}

# 中文安装总结
zh_print_summary_dev() {
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

# 中文 root 警告
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

# 中文远程脚本确认
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

# 检测/安装 Python（通过 uv 管理）
zh_detect_python() {
    # 优先使用 uv 管理的 Python 3.13
    if command -v uv >/dev/null 2>&1; then
        uv_python=$(uv python find 3.13 2>/dev/null)
        if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
            PYTHON_CMD="$uv_python"
            ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            print_success "Python $ver (uv 管理)"
            return 0
        fi

        # 未找到，自动安装
        print_info "正在安装 Python 3.13..."
        if uv python install 3.13; then
            uv_python=$(uv python find 3.13 2>/dev/null)
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                PYTHON_CMD="$uv_python"
                ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
                print_success "Python $ver 安装成功 (uv 管理)"
                return 0
            fi
        fi
        print_error "Python 3.13 安装失败"
    else
        print_error "uv 未安装，无法管理 Python"
    fi

    return 1
}

# 安装 UV（开发者版必需）
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
        print_error "UV 是开发所必需的"
        track_install "uv" "failed"
        return 1
    fi

    # 检查 curl 可用性
    if ! command -v curl >/dev/null 2>&1; then
        print_error "未找到 curl，无法下载 UV 安装脚本"
        print_info "请先安装 curl:"
        print_info "  Ubuntu/Debian: sudo apt install curl"
        print_info "  macOS: brew install curl"
        return 1
    fi

    # 构建安装 URL（支持版本固定）
    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
        print_info "安装 UV 版本: $UV_VERSION"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    # 确认远程脚本执行
    if ! zh_confirm_remote_script "$uv_url" "UV"; then
        print_error "UV 是开发所必需的"
        track_install "uv" "failed"
        return 1
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

# 同步开发依赖
zh_sync_dependencies() {
    project_root=$(get_project_root)
    print_info "项目目录: $project_root"

    cd "$project_root"

    # 关键: 使用 --python 指定检测到的 Python 版本
    print_info "运行 uv sync --all-extras --python $PYTHON_CMD..."
    if uv sync --all-extras --python "$PYTHON_CMD"; then
        print_success "依赖同步完成 (使用 $PYTHON_CMD)"
        return 0
    else
        print_error "依赖同步失败"
        return 1
    fi
}

# 安装 pre-commit hooks
zh_install_precommit() {
    project_root=$(get_project_root)
    cd "$project_root"

    if [ -f ".pre-commit-config.yaml" ]; then
        print_info "安装 pre-commit hooks..."

        if uv run pre-commit install; then
            print_success "pre-commit hooks 安装完成"
            return 0
        else
            print_warning "pre-commit 安装失败，请手动运行: uv run pre-commit install"
            return 0
        fi
    else
        print_info "未找到 .pre-commit-config.yaml，跳过"
    fi

    return 0
}

# 检测 Node.js
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

    version=$(node --version 2>/dev/null)
    major=$(echo "$version" | sed 's/^v//' | cut -d. -f1)

    # 校验是否为纯数字
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
        print_info "使用 pnpm 安装..."
        if pnpm add -g @anthropic-ai/claude-code; then
            print_success "通过 pnpm 安装 Claude Code CLI 成功"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "使用 npm 安装..."
        if npm install -g @anthropic-ai/claude-code >/dev/null 2>&1; then
            print_success "通过 npm 安装 Claude Code CLI 成功"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        print_info "使用 Homebrew 安装..."
        if brew install claude-code >/dev/null 2>&1; then
            print_success "通过 Homebrew 安装 Claude Code CLI 成功"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    print_warning "Claude Code CLI 安装失败"
    print_info "手动安装选项:"
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
        print_info "使用 pnpm 安装..."
        if pnpm add -g @github/copilot; then
            print_success "通过 pnpm 安装 Copilot CLI 成功"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "使用 npm 安装..."
        if npm install -g @github/copilot >/dev/null 2>&1; then
            print_success "通过 npm 安装 Copilot CLI 成功"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        print_info "使用 Homebrew 安装..."
        if brew install copilot-cli >/dev/null 2>&1; then
            print_success "通过 Homebrew 安装 Copilot CLI 成功"
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
            print_success "通过安装脚本安装 Copilot CLI 成功"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    print_warning "Copilot CLI 安装失败"
    print_info "手动安装选项:"
    print_info "  pnpm: pnpm add -g @github/copilot"
    print_info "  brew: brew install copilot-cli"
    print_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    track_install "Copilot CLI" "failed"
    return 1
}

# 安装 LLM CLI 工具
zh_install_llm_clis() {
    print_info "LLM CLI 工具提供本地认证:"
    print_info "  - Claude Code CLI: 使用你的 Claude 订阅"
    print_info "  - Copilot CLI: 使用你的 GitHub Copilot 订阅"

    if ask_yes_no "是否安装 Claude Code CLI?" "n"; then
        zh_install_claude_cli
    fi

    if ask_yes_no "是否安装 GitHub Copilot CLI?" "n"; then
        zh_install_copilot_cli
    fi
}

# 安装 Playwright 浏览器 (Chromium) 及系统依赖 - 开发环境
# 优先使用 uv run，回退到 python 模块
# 返回: 0 成功, 1 失败, 2 跳过
zh_dev_install_playwright_browser() {
    print_info "Playwright 浏览器 (Chromium):"
    print_info "  用途: 浏览器自动化，用于 JavaScript 渲染页面 (Twitter, SPA)"

    project_root=$(get_project_root)
    cd "$project_root"

    # 下载前先征询用户同意
    if ! ask_yes_no "是否下载 Chromium 浏览器？" "y"; then
        print_info "跳过 Playwright 浏览器安装"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

    print_info "正在下载 Chromium 浏览器..."
    browser_installed=false

    # 优先使用 uv run（开发环境使用 .venv）
    if command -v uv >/dev/null 2>&1; then
        if uv run playwright install chromium 2>/dev/null; then
            print_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    # 回退到 Python 模块
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if "$PYTHON_CMD" -m playwright install chromium 2>/dev/null; then
            print_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        print_warning "Playwright 浏览器安装失败"
        print_info "稍后可手动安装: uv run playwright install chromium"
        track_install "Playwright Browser" "failed"
        return 1
    fi

    # 在 Linux 上安装系统依赖（需要 sudo）
    if [ "$(uname)" = "Linux" ]; then
        print_info "Chromium 在 Linux 上需要系统依赖"
        if ask_yes_no "是否安装系统依赖（需要 sudo）？" "y"; then
            print_info "正在安装系统依赖..."

            # Arch Linux: 使用 pacman（playwright install-deps 不支持 Arch）
            if [ -f /etc/arch-release ]; then
                print_info "检测到 Arch Linux，使用 pacman 安装依赖..."
                # Playwright Chromium 核心依赖
                local arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo"
                # 可选字体（提升中文/日文显示）
                local arch_fonts="noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
                if sudo pacman -S --noconfirm --needed $arch_deps $arch_fonts 2>/dev/null; then
                    print_success "系统依赖安装成功"
                    track_install "Playwright Browser" "installed"
                    return 0
                else
                    print_warning "部分依赖安装失败"
                    print_info "可手动安装: sudo pacman -S $arch_deps"
                fi
            # Debian/Ubuntu: 使用 playwright install-deps
            elif command -v apt-get >/dev/null 2>&1; then
                if command -v uv >/dev/null 2>&1; then
                    if uv run playwright install-deps chromium 2>/dev/null; then
                        print_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # 回退到 Python 模块
                if [ -n "$PYTHON_CMD" ]; then
                    if "$PYTHON_CMD" -m playwright install-deps chromium 2>/dev/null; then
                        print_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                print_warning "系统依赖安装失败"
                print_info "可手动安装: sudo playwright install-deps chromium"
            # 其他发行版
            else
                print_warning "未识别的 Linux 发行版"
                print_info "请手动安装 Chromium 依赖"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        else
            print_warning "已跳过系统依赖安装"
            print_info "Chromium 可能无法运行"
            if [ -f /etc/arch-release ]; then
                print_info "稍后安装: sudo pacman -S nss nspr at-spi2-core cups libdrm mesa alsa-lib"
            else
                print_info "稍后安装: sudo playwright install-deps chromium"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        fi
    fi

    track_install "Playwright Browser" "installed"
    return 0
}

# 安装 LibreOffice（可选，用于旧版 Office 文件）
zh_dev_install_libreoffice() {
    print_info "正在检测 LibreOffice..."
    print_info "  用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)"

    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice 已安装: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    if command -v libreoffice >/dev/null 2>&1; then
        version=$(libreoffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice 已安装: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    print_warning "LibreOffice 未安装（可选）"
    print_info "  若未安装，无法转换 .doc/.ppt/.xls 文件"
    print_info "  新版格式 (.docx/.pptx/.xlsx) 无需 LibreOffice"

    if ! ask_yes_no "是否安装 LibreOffice？" "n"; then
        print_info "跳过 LibreOffice 安装"
        track_install "LibreOffice" "skipped"
        return 2
    fi

    print_info "正在安装 LibreOffice..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice >/dev/null 2>&1; then
                    print_success "LibreOffice 通过 Homebrew 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "未找到 Homebrew"
                print_info "请先安装 Homebrew: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "通过 apt 安装（需要 sudo）..."
                if sudo apt update >/dev/null 2>&1 && sudo apt install -y libreoffice >/dev/null 2>&1; then
                    print_success "LibreOffice 通过 apt 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                print_info "通过 dnf 安装（需要 sudo）..."
                if sudo dnf install -y libreoffice >/dev/null 2>&1; then
                    print_success "LibreOffice 通过 dnf 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                print_info "通过 pacman 安装（需要 sudo）..."
                if sudo pacman -S --noconfirm libreoffice-fresh >/dev/null 2>&1; then
                    print_success "LibreOffice 通过 pacman 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "未知的 Linux 发行版"
                print_info "请手动安装 LibreOffice"
            fi
            ;;
    esac

    print_warning "LibreOffice 安装失败"
    print_info "手动安装方式:"
    case "$(uname)" in
        Darwin)
            print_info "  brew install --cask libreoffice"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "  sudo apt install libreoffice"
            else
                print_info "  使用包管理器安装 libreoffice"
            fi
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# 安装 FFmpeg（可选，用于音视频文件处理）
zh_dev_install_ffmpeg() {
    print_info "正在检测 FFmpeg..."
    print_info "  用途: 处理音视频文件 (.mp3, .mp4, .wav 等)"

    if command -v ffmpeg >/dev/null 2>&1; then
        version=$(ffmpeg -version 2>/dev/null | head -n1)
        print_success "FFmpeg 已安装: $version"
        track_install "FFmpeg" "installed"
        return 0
    fi

    print_warning "FFmpeg 未安装（可选）"
    print_info "  若未安装，无法处理音视频文件"

    if ! ask_yes_no "是否安装 FFmpeg？" "n"; then
        print_info "跳过 FFmpeg 安装"
        track_install "FFmpeg" "skipped"
        return 2
    fi

    print_info "正在安装 FFmpeg..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install ffmpeg >/dev/null 2>&1; then
                    print_success "FFmpeg 通过 Homebrew 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                print_error "未找到 Homebrew"
                print_info "请先安装 Homebrew: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "通过 apt 安装（需要 sudo）..."
                if sudo apt update >/dev/null 2>&1 && sudo apt install -y ffmpeg >/dev/null 2>&1; then
                    print_success "FFmpeg 通过 apt 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                print_info "通过 dnf 安装（需要 sudo）..."
                if sudo dnf install -y ffmpeg >/dev/null 2>&1; then
                    print_success "FFmpeg 通过 dnf 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                print_info "通过 pacman 安装（需要 sudo）..."
                if sudo pacman -S --noconfirm ffmpeg >/dev/null 2>&1; then
                    print_success "FFmpeg 通过 pacman 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                print_error "未知的 Linux 发行版"
                print_info "请手动安装 FFmpeg"
            fi
            ;;
    esac

    print_warning "FFmpeg 安装失败"
    print_info "手动安装方式:"
    case "$(uname)" in
        Darwin)
            print_info "  brew install ffmpeg"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "  sudo apt install ffmpeg"
            elif [ -f /etc/fedora-release ]; then
                print_info "  sudo dnf install ffmpeg"
            elif [ -f /etc/arch-release ]; then
                print_info "  sudo pacman -S ffmpeg"
            else
                print_info "  使用包管理器安装 ffmpeg"
            fi
            ;;
    esac
    track_install "FFmpeg" "failed"
    return 1
}

# 打印完成信息
zh_print_completion() {
    project_root=$(get_project_root)

    printf "\n"
    printf "${GREEN}✓${NC} ${BOLD}开发环境配置完成!${NC}\n"
    printf "\n"
    printf "  ${BOLD}激活虚拟环境:${NC}\n"
    printf "    ${YELLOW}source %s/.venv/bin/activate${NC}\n" "$project_root"
    printf "\n"
    printf "  ${BOLD}运行测试:${NC}\n"
    printf "    ${YELLOW}uv run pytest${NC}\n"
    printf "\n"
    printf "  ${BOLD}运行 CLI:${NC}\n"
    printf "    ${YELLOW}uv run markitai --help${NC}\n"
    printf "\n"
}

# ============================================================
# 主逻辑
# ============================================================

main() {
    # 安全检查: root 警告
    zh_warn_if_root

    # 欢迎信息
    zh_print_welcome_dev

    print_header "Markitai 开发环境配置向导"

    # 步骤 1: 检测/安装 UV（开发者版必需，也用于管理 Python）
    print_step 1 5 "检测 UV 包管理器..."
    if ! zh_install_uv; then
        zh_print_summary_dev
        exit 1
    fi

    # 步骤 2: 检测/安装 Python（可通过 uv 自动安装）
    print_step 2 5 "检测 Python..."
    if ! zh_detect_python; then
        exit 1
    fi

    # 步骤 3: 同步依赖（包含所有 extras: browser, claude-agent, copilot）
    print_step 3 5 "同步开发依赖..."
    if ! zh_sync_dependencies; then
        zh_print_summary_dev
        exit 1
    fi
    track_install "Python 依赖" "installed"
    track_install "Claude Agent SDK" "installed"
    track_install "Copilot SDK" "installed"

    # 安装 Playwright 浏览器（SPA/JS 渲染页面需要）
    zh_dev_install_playwright_browser

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    zh_dev_install_libreoffice

    # 安装 FFmpeg（可选，用于音视频文件）
    zh_dev_install_ffmpeg

    # 步骤 4: 安装 pre-commit
    print_step 4 5 "配置 pre-commit..."
    if zh_install_precommit; then
        track_install "pre-commit hooks" "installed"
    fi

    # 步骤 5: 可选组件 - LLM CLI 工具
    print_step 5 5 "可选: LLM CLI 工具"
    if ask_yes_no "是否安装 LLM CLI 工具 (Claude Code / Copilot)?" "n"; then
        zh_install_llm_clis
    else
        print_info "跳过 LLM CLI 安装"
        track_install "Claude Code CLI" "skipped"
        track_install "Copilot CLI" "skipped"
    fi

    # 打印总结
    zh_print_summary_dev

    # 完成
    zh_print_completion
}

# 运行主函数
main
