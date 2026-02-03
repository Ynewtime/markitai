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
    printf "    ${YELLOW}•${NC} Playwright - 浏览器自动化（JS 渲染页面）\n"
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

zh_install_uv() {
    print_info "检查 UV 安装..."

    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        print_success "$version 已安装"
        track_install "uv" "installed"
        return 0
    fi

    print_info "UV 未安装（用于管理 Python 和依赖）"

    if ! ask_yes_no "是否自动安装 UV?" "y"; then
        print_error "UV 是必需的，无法继续"
        track_install "uv" "failed"
        return 1
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

    # 注意: 使用 [browser] 而非 [all] 避免安装不必要的 SDK 包
    # SDK 包 (claude-agent, copilot) 将在用户选择安装 CLI 工具时安装
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[browser]==$MARKITAI_VERSION"
        print_info "安装版本: $MARKITAI_VERSION"
    else
        pkg="markitai[browser]"
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

# 安装 Playwright 浏览器 (Chromium) 及系统依赖
# 安全性: 使用 markitai 虚拟环境中的 playwright 确保使用正确版本
# 返回: 0 成功, 1 失败, 2 跳过
zh_install_playwright_browser() {
    print_info "Playwright 浏览器 (Chromium):"
    print_info "  用途: 浏览器自动化，用于 JavaScript 渲染页面 (Twitter, SPA)"

    # 下载前先征询用户同意
    if ! ask_yes_no "是否下载 Chromium 浏览器？" "y"; then
        print_info "跳过 Playwright 浏览器安装"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

    print_info "正在下载 Chromium 浏览器..."
    browser_installed=false

    # 方法 1: 使用 markitai 的 uv tool 环境中的 playwright（首选）
    # 确保使用与 markitai 依赖相同的 playwright 版本
    # 使用 'uv tool dir' 获取正确路径（兼容 UV_TOOL_DIR, XDG_DATA_HOME）
    markitai_playwright=""
    if command -v uv >/dev/null 2>&1; then
        uv_tool_dir=$(uv tool dir 2>/dev/null)
        if [ -n "$uv_tool_dir" ]; then
            markitai_playwright="$uv_tool_dir/markitai/bin/playwright"
        fi
    fi
    # 如果 uv tool dir 检测失败，回退到默认路径
    if [ -z "$markitai_playwright" ] || [ ! -x "$markitai_playwright" ]; then
        markitai_playwright="$HOME/.local/share/uv/tools/markitai/bin/playwright"
    fi

    if [ -x "$markitai_playwright" ]; then
        if "$markitai_playwright" install chromium 2>/dev/null; then
            print_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    # 方法 2: 回退到 Python 模块（用于 pip/pipx 安装）
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if "$PYTHON_CMD" -m playwright install chromium 2>/dev/null; then
            print_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        print_warning "Playwright 浏览器安装失败"
        print_info "稍后可手动安装: playwright install chromium"
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
                # 方法 1: 使用 markitai 环境中的 playwright
                if [ -x "$markitai_playwright" ]; then
                    if "$markitai_playwright" install-deps chromium 2>/dev/null; then
                        print_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # 方法 2: 回退到 Python 模块
                if [ -n "$PYTHON_CMD" ]; then
                    if "$PYTHON_CMD" -m playwright install-deps chromium 2>/dev/null; then
                        print_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                print_warning "系统依赖安装失败"
                print_info "可手动安装: sudo playwright install-deps chromium"
                print_info "或: sudo apt install libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxdamage1 libgbm1 libpango-1.0-0 libcairo2 libasound2"
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

# 检测 LibreOffice 安装
# LibreOffice 用于转换 .doc, .ppt, .xls 文件
zh_install_libreoffice() {
    print_info "正在检测 LibreOffice..."
    print_info "  用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)"

    # 检测 soffice 命令
    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice 已安装: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    # 检测 libreoffice 命令
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
        return 2  # Skipped
    fi

    print_info "正在安装 LibreOffice..."

    case "$(uname)" in
        Darwin)
            # macOS: 使用 Homebrew
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice; then
                    print_success "LibreOffice 通过 Homebrew 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "未找到 Homebrew"
                print_info "请先安装 Homebrew: https://brew.sh"
                print_info "然后运行: brew install --cask libreoffice"
            fi
            ;;
        Linux)
            # Linux: 使用包管理器
            if [ -f /etc/debian_version ]; then
                # Debian/Ubuntu
                print_info "通过 apt 安装（需要 sudo）..."
                if sudo apt update && sudo apt install -y libreoffice; then
                    print_success "LibreOffice 通过 apt 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                # Fedora
                print_info "通过 dnf 安装（需要 sudo）..."
                if sudo dnf install -y libreoffice; then
                    print_success "LibreOffice 通过 dnf 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                # Arch Linux
                print_info "通过 pacman 安装（需要 sudo）..."
                if sudo pacman -S --noconfirm libreoffice-fresh; then
                    print_success "LibreOffice 通过 pacman 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "未知的 Linux 发行版"
                print_info "请使用包管理器手动安装 LibreOffice"
            fi
            ;;
        *)
            print_error "此平台不支持自动安装"
            print_info "下载地址: https://www.libreoffice.org/download/"
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
            elif [ -f /etc/fedora-release ]; then
                print_info "  sudo dnf install libreoffice"
            elif [ -f /etc/arch-release ]; then
                print_info "  sudo pacman -S libreoffice-fresh"
            else
                print_info "  使用包管理器安装 libreoffice"
            fi
            ;;
        *)
            print_info "  下载地址: https://www.libreoffice.org/download/"
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# 检测 FFmpeg 安装
# FFmpeg 用于处理音视频文件
zh_install_ffmpeg() {
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
                if brew install ffmpeg; then
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
                if sudo apt update && sudo apt install -y ffmpeg; then
                    print_success "FFmpeg 通过 apt 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                print_info "通过 dnf 安装（需要 sudo）..."
                if sudo dnf install -y ffmpeg; then
                    print_success "FFmpeg 通过 dnf 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                print_info "通过 pacman 安装（需要 sudo）..."
                if sudo pacman -S --noconfirm ffmpeg; then
                    print_success "FFmpeg 通过 pacman 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                print_error "未知的 Linux 发行版"
                print_info "请手动安装 FFmpeg"
            fi
            ;;
        *)
            print_error "此平台不支持自动安装"
            print_info "下载地址: https://ffmpeg.org/download.html"
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
        *)
            print_info "  下载地址: https://ffmpeg.org/download.html"
            ;;
    esac
    track_install "FFmpeg" "failed"
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

    if ! command -v markitai >/dev/null 2>&1; then
        return 0
    fi

    local config_path="$HOME/.markitai/config.json"
    local yes_flag=""

    # Check if config exists and ask user (using ask_yes_no for piped execution)
    if [ -f "$config_path" ]; then
        if ask_yes_no "$config_path 已存在，是否覆盖？" "n"; then
            yes_flag="--yes"
        else
            print_info "保留现有配置"
            return 0
        fi
    fi

    if markitai config init $yes_flag 2>/dev/null; then
        print_success "配置初始化完成"
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

    # 步骤 1: 检测/安装 UV（用于管理 Python 和依赖）
    print_step 1 5 "检测 UV 包管理器..."
    if ! zh_install_uv; then
        zh_print_summary
        exit 1
    fi

    # 步骤 2: 检测/安装 Python（通过 uv 自动安装）
    print_step 2 5 "检测 Python..."
    if ! zh_detect_python; then
        exit 1
    fi

    # 步骤 3: 安装 markitai
    print_step 3 5 "安装 markitai..."
    if ! zh_install_markitai; then
        zh_print_summary
        exit 1
    fi

    # 安装 Playwright 浏览器 (SPA/JS 渲染页面需要)
    zh_install_playwright_browser

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    zh_install_libreoffice

    # 安装 FFmpeg（可选，用于音视频文件）
    zh_install_ffmpeg

    # 步骤 4: 可选 - LLM CLI 工具
    print_step 4 5 "可选: LLM CLI 工具"
    print_info "LLM CLI 工具为 AI 提供商提供本地认证"
    if ask_yes_no "是否安装 Claude Code CLI?" "n"; then
        if zh_install_claude_cli; then
            # 安装 Claude Agent SDK 以支持编程式访问
            lib_install_markitai_extra "claude-agent"
        fi
    else
        track_install "Claude Code CLI" "skipped"
    fi
    if ask_yes_no "是否安装 GitHub Copilot CLI?" "n"; then
        if zh_install_copilot_cli; then
            # 安装 Copilot SDK 以支持编程式访问
            lib_install_markitai_extra "copilot"
        fi
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
