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
# 中文输出覆盖（使用 clack 风格组件）
# ============================================================

# 中文欢迎信息（用户版）- 使用 clack_intro 和 clack_note
zh_print_welcome_user() {
    clack_intro "欢迎使用 Markitai 安装向导!"
    clack_note "安装内容" \
        "${GREEN}•${NC} markitai - 支持 LLM 的 Markdown 转换器" \
        "" \
        "${BOLD}可选组件:${NC}" \
        "${YELLOW}•${NC} Playwright - 浏览器自动化（JS 渲染页面）" \
        "${YELLOW}•${NC} Claude Code CLI - 使用 Claude 订阅" \
        "${YELLOW}•${NC} Copilot CLI - 使用 GitHub Copilot 订阅" \
        "" \
        "随时按 ${BOLD}Ctrl+C${NC} 取消"
}

# 中文安装总结 - 使用 clack 风格
zh_print_summary() {
    clack_section "安装总结"

    # 已安装
    if [ -n "$INSTALLED_COMPONENTS" ]; then
        clack_log "${GREEN}已安装:${NC}"
        echo "$INSTALLED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && clack_success "$comp"
        done
    fi

    # 已跳过
    if [ -n "$SKIPPED_COMPONENTS" ]; then
        clack_log "${YELLOW}已跳过:${NC}"
        echo "$SKIPPED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && clack_skip "$comp"
        done
    fi

    # 安装失败
    if [ -n "$FAILED_COMPONENTS" ]; then
        clack_log "${RED}安装失败:${NC}"
        echo "$FAILED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && clack_error "$comp"
        done
    fi

    clack_log ""
    clack_log "${BOLD}文档:${NC} https://markitai.ynewtime.com"
    clack_log "${BOLD}问题反馈:${NC} https://github.com/Ynewtime/markitai/issues"
    return 0
}

zh_warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        clack_section "警告: 正在以 root 身份运行"
        clack_warn "以 root 身份运行安装脚本存在以下风险:"
        clack_log "  1. PATH 劫持: ~/.local/bin 可能被其他用户写入"
        clack_log "  2. 远程代码执行风险被放大"
        clack_log ""
        clack_info "建议: 使用普通用户身份运行此脚本"

        if ! clack_confirm "是否继续以 root 身份运行?" "n"; then
            clack_cancel "退出。请使用普通用户身份运行。"
            exit 1
        fi
    fi
    return 0
}

zh_confirm_remote_script() {
    script_url="$1"
    script_name="$2"

    clack_section "警告: 即将执行远程脚本"
    clack_log "来源: $script_url"
    clack_log "用途: 安装 $script_name"
    clack_log ""
    clack_warn "此操作将从互联网下载并执行代码。"
    clack_info "请确保您信任该来源。"

    if clack_confirm "确认执行?" "n"; then
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
            clack_success "Python $ver (uv 管理)"
            return 0
        fi

        # 未找到，自动安装
        clack_info "正在安装 Python 3.13..."
        if uv python install 3.13; then
            uv_python=$(uv python find 3.13 2>/dev/null)
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                PYTHON_CMD="$uv_python"
                ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
                clack_success "Python $ver 安装成功 (uv 管理)"
                return 0
            fi
        fi
        clack_error "Python 3.13 安装失败"
    else
        clack_error "uv 未安装，无法管理 Python"
    fi

    return 1
}

zh_install_uv() {
    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        clack_success "$version 已安装"
        track_install "uv" "installed"
        return 0
    fi

    clack_info "UV 未安装（用于管理 Python 和依赖）"

    if ! clack_confirm "是否自动安装 UV?" "y"; then
        clack_error "UV 是必需的，无法继续"
        track_install "uv" "failed"
        return 1
    fi

    if ! command -v curl >/dev/null 2>&1; then
        clack_error "未找到 curl，无法下载 UV 安装脚本"
        clack_info "请先安装 curl:"
        clack_log "  Ubuntu/Debian: sudo apt install curl"
        clack_log "  macOS: brew install curl"
        clack_log "  或手动安装 UV: https://docs.astral.sh/uv/getting-started/installation/"
        return 1
    fi

    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
        clack_info "安装 UV 版本: $UV_VERSION"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    if ! zh_confirm_remote_script "$uv_url" "UV"; then
        clack_skip "跳过 UV 安装"
        track_install "uv" "skipped"
        return 2
    fi

    clack_spinner "正在安装 UV..." curl -LsSf "$uv_url" | sh
    _uv_status=$?

    if [ $_uv_status -eq 0 ]; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1)
            clack_success "$version 安装成功"
            track_install "uv" "installed"
            return 0
        else
            clack_warn "UV 已安装，但需要重新加载 shell"
            clack_info "请运行: source ~/.bashrc 或重新打开终端"
            clack_info "然后重新运行此脚本"
            track_install "uv" "installed"
            return 1
        fi
    else
        clack_error "UV 安装失败"
        clack_info "手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
        track_install "uv" "failed"
        return 1
    fi
}

zh_install_markitai() {
    # 注意: 使用 [browser] 而非 [all] 避免安装不必要的 SDK 包
    # SDK 包 (claude-agent, copilot) 将在用户选择安装 CLI 工具时安装
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[browser]==$MARKITAI_VERSION"
        clack_info "安装版本: $MARKITAI_VERSION"
    else
        pkg="markitai[browser]"
    fi

    if command -v uv >/dev/null 2>&1; then
        # 使用 --upgrade 确保安装最新版本
        clack_spinner "正在安装 markitai..." uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade
        if [ $? -eq 0 ]; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "已安装")
            clack_success "markitai $version 安装成功"
            clack_info "已安装到 ~/.local/bin (使用 $PYTHON_CMD)"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    if command -v pipx >/dev/null 2>&1; then
        # 使用 --force 确保安装最新版本
        clack_spinner "正在安装 markitai..." pipx install "$pkg" --python "$PYTHON_CMD" --force
        if [ $? -eq 0 ]; then
            version=$(markitai --version 2>/dev/null || echo "已安装")
            clack_success "markitai $version 安装成功"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # 使用 --upgrade 确保安装最新版本
    clack_spinner "正在安装 markitai..." "$PYTHON_CMD" -m pip install --user --upgrade "$pkg"
    if [ $? -eq 0 ]; then
        export PATH="$HOME/.local/bin:$PATH"
        version=$(markitai --version 2>/dev/null || echo "已安装")
        clack_success "markitai $version 安装成功"
        clack_warn "可能需要将 ~/.local/bin 添加到 PATH"
        track_install "markitai" "installed"
        return 0
    fi

    clack_error "markitai 安装失败"
    clack_info "请手动安装: uv tool install markitai --python $PYTHON_CMD"
    track_install "markitai" "failed"
    return 1
}

# 安装 Playwright 浏览器 (Chromium) 及系统依赖
# 安全性: 使用 markitai 虚拟环境中的 playwright 确保使用正确版本
# 返回: 0 成功, 1 失败, 2 跳过
zh_install_playwright_browser() {
    clack_log "Playwright 浏览器 (Chromium):"
    clack_info "用途: 浏览器自动化，用于 JavaScript 渲染页面 (Twitter, SPA)"

    # 先检测是否已安装
    if lib_detect_playwright_browser; then
        clack_success "Playwright Chromium 已安装"
        track_install "Playwright Browser" "installed"
        return 0
    fi

    # 下载前先征询用户同意
    if ! clack_confirm "是否下载 Chromium 浏览器？" "y"; then
        clack_skip "跳过 Playwright 浏览器安装"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

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

    # 使用 clack_spinner 显示下载进度
    if [ -x "$markitai_playwright" ]; then
        clack_spinner "正在下载 Chromium 浏览器..." "$markitai_playwright" install chromium
        if [ $? -eq 0 ]; then
            clack_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    # 方法 2: 回退到 Python 模块（用于 pip/pipx 安装）
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        clack_spinner "正在下载 Chromium 浏览器..." "$PYTHON_CMD" -m playwright install chromium
        if [ $? -eq 0 ]; then
            clack_success "Chromium 浏览器下载成功"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        clack_warn "Playwright 浏览器安装失败"
        clack_info "稍后可手动安装: playwright install chromium"
        track_install "Playwright Browser" "failed"
        return 1
    fi

    # 在 Linux 上安装系统依赖（需要 sudo）
    if [ "$(uname)" = "Linux" ]; then
        clack_info "Chromium 在 Linux 上需要系统依赖"
        if clack_confirm "是否安装系统依赖（需要 sudo）？" "y"; then
            # Arch Linux: 使用 pacman（playwright install-deps 不支持 Arch）
            if [ -f /etc/arch-release ]; then
                clack_info "检测到 Arch Linux，使用 pacman 安装依赖..."
                # Playwright Chromium 核心依赖
                arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo"
                # 可选字体（提升中文/日文显示）
                arch_fonts="noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
                clack_spinner "正在安装系统依赖..." sudo pacman -S --noconfirm --needed $arch_deps $arch_fonts
                if [ $? -eq 0 ]; then
                    clack_success "系统依赖安装成功"
                    track_install "Playwright Browser" "installed"
                    return 0
                else
                    clack_warn "部分依赖安装失败"
                    clack_info "可手动安装: sudo pacman -S $arch_deps"
                fi
            # Debian/Ubuntu: 使用 playwright install-deps
            elif command -v apt-get >/dev/null 2>&1; then
                # 方法 1: 使用 markitai 环境中的 playwright
                if [ -x "$markitai_playwright" ]; then
                    clack_spinner "正在安装系统依赖..." "$markitai_playwright" install-deps chromium
                    if [ $? -eq 0 ]; then
                        clack_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # 方法 2: 回退到 Python 模块
                if [ -n "$PYTHON_CMD" ]; then
                    clack_spinner "正在安装系统依赖..." "$PYTHON_CMD" -m playwright install-deps chromium
                    if [ $? -eq 0 ]; then
                        clack_success "系统依赖安装成功"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                clack_warn "系统依赖安装失败"
                clack_info "可手动安装: sudo playwright install-deps chromium"
                clack_info "或: sudo apt install libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxdamage1 libgbm1 libpango-1.0-0 libcairo2 libasound2"
            # 其他发行版
            else
                clack_warn "未识别的 Linux 发行版"
                clack_info "请手动安装 Chromium 依赖"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        else
            clack_warn "已跳过系统依赖安装"
            clack_info "Chromium 可能无法运行"
            if [ -f /etc/arch-release ]; then
                clack_info "稍后安装: sudo pacman -S nss nspr at-spi2-core cups libdrm mesa alsa-lib"
            else
                clack_info "稍后安装: sudo playwright install-deps chromium"
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
    clack_log "LibreOffice (可选):"
    clack_info "用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)"
    clack_info "新版格式 (.docx/.pptx/.xlsx) 无需 LibreOffice"

    # 检测 soffice 命令
    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        clack_success "LibreOffice 已安装: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    # 检测 libreoffice 命令
    if command -v libreoffice >/dev/null 2>&1; then
        version=$(libreoffice --version 2>/dev/null | head -n1)
        clack_success "LibreOffice 已安装: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    if ! clack_confirm "是否安装 LibreOffice？" "n"; then
        clack_skip "跳过 LibreOffice 安装"
        track_install "LibreOffice" "skipped"
        return 2  # Skipped
    fi

    case "$(uname)" in
        Darwin)
            # macOS: 使用 Homebrew
            if command -v brew >/dev/null 2>&1; then
                clack_spinner "正在安装 LibreOffice..." brew install --cask libreoffice
                if [ $? -eq 0 ]; then
                    clack_success "LibreOffice 通过 Homebrew 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                clack_error "未找到 Homebrew"
                clack_info "请先安装 Homebrew: https://brew.sh"
                clack_info "然后运行: brew install --cask libreoffice"
            fi
            ;;
        Linux)
            # Linux: 使用包管理器
            if [ -f /etc/debian_version ]; then
                # Debian/Ubuntu
                clack_spinner "正在安装 LibreOffice..." sh -c "sudo apt update >/dev/null 2>&1 && sudo apt install -y libreoffice >/dev/null 2>&1"
                if [ $? -eq 0 ]; then
                    clack_success "LibreOffice 通过 apt 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                # Fedora
                clack_spinner "正在安装 LibreOffice..." sudo dnf install -y libreoffice
                if [ $? -eq 0 ]; then
                    clack_success "LibreOffice 通过 dnf 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                # Arch Linux
                clack_spinner "正在安装 LibreOffice..." sudo pacman -S --noconfirm libreoffice-fresh
                if [ $? -eq 0 ]; then
                    clack_success "LibreOffice 通过 pacman 安装成功"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                clack_error "未知的 Linux 发行版"
                clack_info "请使用包管理器手动安装 LibreOffice"
            fi
            ;;
        *)
            clack_error "此平台不支持自动安装"
            clack_info "下载地址: https://www.libreoffice.org/download/"
            ;;
    esac

    clack_warn "LibreOffice 安装失败"
    clack_info "手动安装方式:"
    case "$(uname)" in
        Darwin)
            clack_log "  brew install --cask libreoffice"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_log "  sudo apt install libreoffice"
            elif [ -f /etc/fedora-release ]; then
                clack_log "  sudo dnf install libreoffice"
            elif [ -f /etc/arch-release ]; then
                clack_log "  sudo pacman -S libreoffice-fresh"
            else
                clack_log "  使用包管理器安装 libreoffice"
            fi
            ;;
        *)
            clack_log "  下载地址: https://www.libreoffice.org/download/"
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# 检测 FFmpeg 安装
# FFmpeg 用于处理音视频文件
zh_install_ffmpeg() {
    clack_log "FFmpeg (可选):"
    clack_info "用途: 处理音视频文件 (.mp3, .mp4, .wav 等)"

    if command -v ffmpeg >/dev/null 2>&1; then
        version=$(ffmpeg -version 2>/dev/null | head -n1)
        clack_success "FFmpeg 已安装: $version"
        track_install "FFmpeg" "installed"
        return 0
    fi

    if ! clack_confirm "是否安装 FFmpeg？" "n"; then
        clack_skip "跳过 FFmpeg 安装"
        track_install "FFmpeg" "skipped"
        return 2
    fi

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                clack_spinner "正在安装 FFmpeg..." brew install ffmpeg
                if [ $? -eq 0 ]; then
                    clack_success "FFmpeg 通过 Homebrew 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                clack_error "未找到 Homebrew"
                clack_info "请先安装 Homebrew: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_spinner "正在安装 FFmpeg..." sh -c "sudo apt update >/dev/null 2>&1 && sudo apt install -y ffmpeg >/dev/null 2>&1"
                if [ $? -eq 0 ]; then
                    clack_success "FFmpeg 通过 apt 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                clack_spinner "正在安装 FFmpeg..." sudo dnf install -y ffmpeg
                if [ $? -eq 0 ]; then
                    clack_success "FFmpeg 通过 dnf 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                clack_spinner "正在安装 FFmpeg..." sudo pacman -S --noconfirm ffmpeg
                if [ $? -eq 0 ]; then
                    clack_success "FFmpeg 通过 pacman 安装成功"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                clack_error "未知的 Linux 发行版"
                clack_info "请手动安装 FFmpeg"
            fi
            ;;
        *)
            clack_error "此平台不支持自动安装"
            clack_info "下载地址: https://ffmpeg.org/download.html"
            ;;
    esac

    clack_warn "FFmpeg 安装失败"
    clack_info "手动安装方式:"
    case "$(uname)" in
        Darwin)
            clack_log "  brew install ffmpeg"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_log "  sudo apt install ffmpeg"
            elif [ -f /etc/fedora-release ]; then
                clack_log "  sudo dnf install ffmpeg"
            elif [ -f /etc/arch-release ]; then
                clack_log "  sudo pacman -S ffmpeg"
            else
                clack_log "  使用包管理器安装 ffmpeg"
            fi
            ;;
        *)
            clack_log "  下载地址: https://ffmpeg.org/download.html"
            ;;
    esac
    track_install "FFmpeg" "failed"
    return 1
}

zh_detect_node() {
    if ! command -v node >/dev/null 2>&1; then
        clack_error "未找到 Node.js"
        clack_warn "请安装 Node.js 18+:"
        clack_info "官网下载: https://nodejs.org/"
        clack_info "使用 nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
        clack_info "使用 fnm: curl -fsSL https://fnm.vercel.app/install | bash"
        clack_info "Ubuntu/Debian: sudo apt install nodejs npm"
        clack_info "macOS: brew install node"
        return 1
    fi

    # Get version and strip CR (Windows CRLF compatibility)
    version=$(node --version 2>/dev/null | tr -d '\r')

    # Check if version is empty
    if [ -z "$version" ]; then
        clack_warn "无法获取 Node 版本 (输出为空)"
        return 1
    fi

    # Extract major version number
    major=$(printf '%s' "$version" | sed 's/^v//' | cut -d. -f1)

    case "$major" in
        ''|*[!0-9]*)
            clack_warn "无法解析 Node 版本: $version"
            return 1
            ;;
    esac

    if [ "$major" -ge 18 ]; then
        clack_success "Node.js $version 已安装"
        return 0
    else
        clack_warn "Node.js $version 版本较低，建议 18+"
        return 0
    fi
}

# 安装 Claude Code CLI
zh_install_claude_cli() {
    # 检查是否已安装
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        clack_success "Claude Code CLI 已安装: $version"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # 优先使用 npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        clack_info "通过 pnpm 安装..."
        clack_spinner "正在安装 Claude Code CLI..." pnpm add -g @anthropic-ai/claude-code
        if [ $? -eq 0 ]; then
            clack_success "Claude Code CLI 安装成功 (pnpm)"
            clack_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        clack_info "通过 npm 安装..."
        clack_spinner "正在安装 Claude Code CLI..." npm install -g @anthropic-ai/claude-code
        if [ $? -eq 0 ]; then
            clack_success "Claude Code CLI 安装成功 (npm)"
            clack_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew
    if command -v brew >/dev/null 2>&1; then
        clack_info "通过 Homebrew 安装..."
        clack_spinner "正在安装 Claude Code CLI..." brew install claude-code
        if [ $? -eq 0 ]; then
            clack_success "Claude Code CLI 安装成功 (Homebrew)"
            clack_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    clack_warn "Claude Code CLI 安装失败"
    clack_info "手动安装方式:"
    clack_log "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    clack_log "  brew: brew install claude-code"
    clack_log "  文档: https://code.claude.com/docs/en/setup"
    track_install "Claude Code CLI" "failed"
    return 1
}

# 安装 GitHub Copilot CLI
zh_install_copilot_cli() {
    # 检查是否已安装
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        clack_success "Copilot CLI 已安装: $version"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # 优先使用 npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        clack_info "通过 pnpm 安装..."
        clack_spinner "正在安装 Copilot CLI..." pnpm add -g @github/copilot
        if [ $? -eq 0 ]; then
            clack_success "Copilot CLI 安装成功 (pnpm)"
            clack_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        clack_info "通过 npm 安装..."
        clack_spinner "正在安装 Copilot CLI..." npm install -g @github/copilot
        if [ $? -eq 0 ]; then
            clack_success "Copilot CLI 安装成功 (npm)"
            clack_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew
    if command -v brew >/dev/null 2>&1; then
        clack_info "通过 Homebrew 安装..."
        clack_spinner "正在安装 Copilot CLI..." brew install copilot-cli
        if [ $? -eq 0 ]; then
            clack_success "Copilot CLI 安装成功 (Homebrew)"
            clack_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # 备选: 安装脚本 (需要确认)
    copilot_url="https://gh.io/copilot-install"
    if zh_confirm_remote_script "$copilot_url" "GitHub Copilot CLI"; then
        clack_info "尝试安装脚本..."
        clack_spinner "正在安装 Copilot CLI..." curl -fsSL "$copilot_url" | bash
        if [ $? -eq 0 ]; then
            clack_success "Copilot CLI 安装成功"
            clack_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    clack_warn "Copilot CLI 安装失败"
    clack_info "手动安装方式:"
    clack_log "  pnpm: pnpm add -g @github/copilot"
    clack_log "  brew: brew install copilot-cli"
    clack_log "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    track_install "Copilot CLI" "failed"
    return 1
}

zh_init_config() {
    clack_info "初始化配置..."

    if ! command -v markitai >/dev/null 2>&1; then
        return 0
    fi

    config_path="$HOME/.markitai/config.json"
    yes_flag=""

    # Check if config exists and ask user (using clack_confirm for piped execution)
    if [ -f "$config_path" ]; then
        if clack_confirm "$config_path 已存在，是否覆盖？" "n"; then
            yes_flag="--yes"
        else
            clack_info "保留现有配置"
            return 0
        fi
    fi

    if markitai config init $yes_flag 2>/dev/null; then
        clack_success "配置初始化完成"
    fi
    return 0
}

zh_print_completion() {
    clack_log ""
    clack_log "${BOLD}开始使用:${NC}"
    clack_log "  ${CYAN}markitai -I${NC}          交互模式"
    clack_log "  ${CYAN}markitai file.pdf${NC}   转换文件"
    clack_log "  ${CYAN}markitai --help${NC}     显示所有选项"
}

# ============================================================
# 主逻辑
# ============================================================

main() {
    # 欢迎信息
    zh_print_welcome_user

    # 安全检查: root 警告
    zh_warn_if_root

    # 步骤 1: 检测/安装 UV（用于管理 Python 和依赖）
    clack_section "检测 UV 包管理器"
    if ! zh_install_uv; then
        zh_print_summary
        clack_cancel "安装失败"
        exit 1
    fi

    # 步骤 2: 检测/安装 Python（通过 uv 自动安装）
    clack_section "检测 Python"
    if ! zh_detect_python; then
        zh_print_summary
        clack_cancel "安装失败"
        exit 1
    fi

    # 步骤 3: 安装 markitai
    clack_section "安装 markitai"
    if ! zh_install_markitai; then
        zh_print_summary
        clack_cancel "安装失败"
        exit 1
    fi

    # 可选组件
    clack_section "可选组件"

    # 安装 Playwright 浏览器 (SPA/JS 渲染页面需要)
    zh_install_playwright_browser

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    zh_install_libreoffice

    # 安装 FFmpeg（可选，用于音视频文件）
    zh_install_ffmpeg

    # 步骤 4: 可选 - LLM CLI 工具
    clack_section "LLM CLI 工具 (可选)"
    clack_info "LLM CLI 工具为 AI 提供商提供本地认证"

    # Claude Code CLI - 先检测再询问
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        clack_success "Claude Code CLI 已安装: $version"
        track_install "Claude Code CLI" "installed"
    elif clack_confirm "是否安装 Claude Code CLI?" "n"; then
        if zh_install_claude_cli; then
            lib_install_markitai_extra "claude-agent"
        fi
    else
        clack_skip "跳过 Claude Code CLI"
        track_install "Claude Code CLI" "skipped"
    fi

    # Copilot CLI - 先检测再询问
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        clack_success "Copilot CLI 已安装: $version"
        track_install "Copilot CLI" "installed"
    elif clack_confirm "是否安装 GitHub Copilot CLI?" "n"; then
        if zh_install_copilot_cli; then
            lib_install_markitai_extra "copilot"
        fi
    else
        clack_skip "跳过 Copilot CLI"
        track_install "Copilot CLI" "skipped"
    fi

    # 初始化配置
    zh_init_config

    # 打印总结
    zh_print_summary

    # 完成
    zh_print_completion

    # 结束
    clack_outro "配置完成!"
}

# 运行主函数
main
