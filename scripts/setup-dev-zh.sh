#!/bin/sh
# Markitai 环境配置脚本 (开发者版)
# 支持 bash/zsh/dash 等 POSIX 兼容 shell

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 辅助函数
print_header() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}%s${NC}\n" "$1"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
}

print_step() {
    printf "${BLUE}[%s/%s]${NC} %s\n" "$1" "$2" "$3"
}

print_success() {
    printf "  ${GREEN}✓${NC} %s\n" "$1"
}

print_error() {
    printf "  ${RED}✗${NC} %s\n" "$1"
}

print_info() {
    printf "  ${YELLOW}→${NC} %s\n" "$1"
}

print_warning() {
    printf "  ${YELLOW}!${NC} %s\n" "$1"
}

# 询问用户 (默认值作为第二个参数)
ask_yes_no() {
    prompt="$1"
    default="$2"

    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n] "
    else
        prompt="$prompt [y/N] "
    fi

    printf "  ${YELLOW}?${NC} %s" "$prompt"
    read -r answer

    if [ -z "$answer" ]; then
        answer="$default"
    fi

    case "$answer" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

# 获取脚本所在目录的父目录 (项目根目录)
get_project_root() {
    # 获取脚本目录
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    # 返回父目录
    dirname "$script_dir"
}

# 检测 Python (需要 3.11-3.13，不支持 3.14+)
detect_python() {
    print_step 1 5 "检测 Python..."

    # 尝试不同的 Python 命令 (优先使用 3.11-3.13)
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
            major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

            # 检查版本范围: 3.11 <= version < 3.14
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
                PYTHON_CMD="$cmd"
                print_success "Python $version 已安装 ($cmd)"
                return 0
            elif [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
                print_warning "Python $version 检测到，但 onnxruntime 不支持 Python 3.14+"
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

# 检测/安装 UV
detect_uv() {
    print_step 2 5 "检测 UV 包管理器..."

    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        print_success "$version 已安装"
        return 0
    fi

    print_error "UV 未安装"

    if ask_yes_no "是否自动安装 UV?" "y"; then
        print_info "正在安装 UV..."

        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            # 刷新 PATH
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

            if command -v uv >/dev/null 2>&1; then
                version=$(uv --version 2>/dev/null | head -n1)
                print_success "$version 安装成功"
                return 0
            else
                print_warning "UV 已安装，但需要重新加载 shell"
                print_info "请运行: source ~/.bashrc 或重新打开终端"
                print_info "然后重新运行此脚本"
                return 1
            fi
        else
            print_error "UV 安装失败"
            print_info "手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
            return 1
        fi
    else
        print_error "UV 是开发所必需的"
        return 1
    fi
}

# 同步开发依赖
sync_dependencies() {
    print_step 3 5 "同步开发依赖..."

    project_root=$(get_project_root)
    print_info "项目目录: $project_root"

    cd "$project_root"

    print_info "运行 uv sync --all-extras..."
    if uv sync --all-extras; then
        print_success "依赖同步完成"
        return 0
    else
        print_error "依赖同步失败"
        return 1
    fi
}

# 安装 pre-commit hooks
install_precommit() {
    print_step 4 5 "配置 pre-commit..."

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

# 可选组件
install_optional() {
    print_step 5 5 "可选组件"

    if ask_yes_no "是否安装浏览器自动化支持 (agent-browser)?" "n"; then
        install_agent_browser
    else
        print_info "跳过 agent-browser 安装"
    fi
}

# 检测 Node.js
detect_nodejs() {
    print_info "检测 Node.js..."

    if command -v node >/dev/null 2>&1; then
        version=$(node --version 2>/dev/null)
        major=$(echo "$version" | sed 's/v//' | cut -d. -f1)

        if [ "$major" -ge 18 ]; then
            print_success "Node.js $version 已安装"
            return 0
        else
            print_warning "Node.js $version 版本较低，建议 18+"
            return 0
        fi
    fi

    print_error "未找到 Node.js"
    printf "\n"
    print_warning "请安装 Node.js 18+:"
    print_info "官网下载: https://nodejs.org/"
    print_info "使用 nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
    print_info "使用 fnm: curl -fsSL https://fnm.vercel.app/install | bash"
    print_info "Ubuntu/Debian: sudo apt install nodejs npm"
    print_info "macOS: brew install node"
    return 1
}

# 安装 agent-browser
install_agent_browser() {
    if ! detect_nodejs; then
        print_warning "跳过 agent-browser 安装 (需要 Node.js)"
        return 1
    fi

    print_info "正在安装 agent-browser..."

    if npm install -g agent-browser; then
        print_success "agent-browser 安装成功"

        # 检测操作系统
        os_type=$(uname -s)

        if ask_yes_no "是否下载 Chromium 浏览器?" "y"; then
            print_info "正在下载 Chromium..."

            if [ "$os_type" = "Linux" ]; then
                # Linux 需要安装系统依赖
                if ask_yes_no "是否同时安装系统依赖 (需要 sudo)?" "y"; then
                    agent-browser install --with-deps
                else
                    agent-browser install
                fi
            else
                agent-browser install
            fi

            print_success "Chromium 下载完成"
        fi

        return 0
    else
        print_error "agent-browser 安装失败"
        print_info "请手动安装: npm install -g agent-browser"
        return 1
    fi
}

# 打印完成信息
print_completion() {
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

# 主函数
main() {
    print_header "Markitai 开发环境配置向导"

    # 检测 Python
    if ! detect_python; then
        exit 1
    fi

    # 检测/安装 UV
    if ! detect_uv; then
        exit 1
    fi

    # 同步依赖
    if ! sync_dependencies; then
        exit 1
    fi

    # 安装 pre-commit
    install_precommit

    # 可选组件
    install_optional

    # 完成
    print_completion
}

# 运行主函数
main
