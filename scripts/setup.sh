#!/bin/sh
# Markitai Setup Script (User Edition)
# Supports bash/zsh/dash and other POSIX-compatible shells

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Helper functions
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

# Ask user (default value as second parameter)
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

# Detect Python (requires 3.11-3.13, 3.14+ not supported)
detect_python() {
    print_step 1 4 "Detecting Python..."

    # Try different Python commands (prefer 3.11-3.13)
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
            major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

            # Check version range: 3.11 <= version < 3.14
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
                PYTHON_CMD="$cmd"
                print_success "Python $version installed ($cmd)"
                return 0
            elif [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
                print_warning "Python $version detected, but onnxruntime doesn't support Python 3.14+"
            fi
        fi
    done

    print_error "Python 3.11-3.13 not found"
    printf "\n"
    print_warning "Please install Python 3.13 (recommended) or 3.11/3.12:"
    print_info "Download: https://www.python.org/downloads/"
    print_info "Ubuntu/Debian: sudo apt install python3.13"
    print_info "macOS: brew install python@3.13"
    print_info "pyenv: pyenv install 3.13"
    print_info "Note: onnxruntime doesn't support Python 3.14 yet"
    return 1
}

# Detect/install UV
detect_uv() {
    print_step 2 4 "Detecting UV package manager..."

    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        print_success "$version installed"
        return 0
    fi

    print_error "UV not installed"

    if ask_yes_no "Install UV automatically?" "y"; then
        print_info "Installing UV..."

        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            # Refresh PATH
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

            if command -v uv >/dev/null 2>&1; then
                version=$(uv --version 2>/dev/null | head -n1)
                print_success "$version installed successfully"
                return 0
            else
                print_warning "UV installed, but shell needs to be reloaded"
                print_info "Run: source ~/.bashrc or restart terminal"
                print_info "Then run this script again"
                return 1
            fi
        else
            print_error "UV installation failed"
            print_info "Manual install: curl -LsSf https://astral.sh/uv/install.sh | sh"
            return 1
        fi
    else
        print_info "Skipping UV installation"
        print_warning "markitai recommends using UV for installation"
        return 1
    fi
}

# Install markitai
install_markitai() {
    print_step 3 4 "Installing markitai..."

    print_info "Installing..."

    # Prefer uv tool install (recommended, installs to ~/.local/bin)
    if command -v uv >/dev/null 2>&1; then
        if uv tool install "markitai[all]" 2>/dev/null; then
            # Ensure ~/.local/bin is in PATH
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            print_info "Installed to ~/.local/bin"
            return 0
        fi
    fi

    # Fallback to pipx
    if command -v pipx >/dev/null 2>&1; then
        if pipx install "markitai[all]"; then
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            return 0
        fi
    fi

    # Fallback to pip --user
    if command -v pip3 >/dev/null 2>&1; then
        if pip3 install --user "markitai[all]"; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            print_warning "You may need to add ~/.local/bin to PATH"
            return 0
        fi
    elif command -v pip >/dev/null 2>&1; then
        if pip install --user "markitai[all]"; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            print_warning "You may need to add ~/.local/bin to PATH"
            return 0
        fi
    fi

    print_error "markitai installation failed"
    print_info "Manual install: uv tool install markitai"
    return 1
}

# Optional components
install_optional() {
    print_step 4 4 "Optional components"

    if ask_yes_no "Install browser automation support (agent-browser)?" "n"; then
        install_agent_browser
    else
        print_info "Skipping agent-browser installation"
    fi
}

# Detect Node.js
detect_nodejs() {
    print_info "Detecting Node.js..."

    if command -v node >/dev/null 2>&1; then
        version=$(node --version 2>/dev/null)
        major=$(echo "$version" | sed 's/v//' | cut -d. -f1)

        if [ "$major" -ge 18 ]; then
            print_success "Node.js $version installed"
            return 0
        else
            print_warning "Node.js $version is outdated, 18+ recommended"
            return 0
        fi
    fi

    print_error "Node.js not found"
    printf "\n"
    print_warning "Please install Node.js 18+:"
    print_info "Download: https://nodejs.org/"
    print_info "nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
    print_info "fnm: curl -fsSL https://fnm.vercel.app/install | bash"
    print_info "Ubuntu/Debian: sudo apt install nodejs npm"
    print_info "macOS: brew install node"
    return 1
}

# Install agent-browser
install_agent_browser() {
    if ! detect_nodejs; then
        print_warning "Skipping agent-browser installation (requires Node.js)"
        return 1
    fi

    print_info "Installing agent-browser..."

    if npm install -g agent-browser; then
        print_success "agent-browser installed successfully"

        # Detect OS
        os_type=$(uname -s)

        if ask_yes_no "Download Chromium browser?" "y"; then
            print_info "Downloading Chromium..."

            if [ "$os_type" = "Linux" ]; then
                # Linux needs system dependencies
                if ask_yes_no "Also install system dependencies (requires sudo)?" "y"; then
                    agent-browser install --with-deps
                else
                    agent-browser install
                fi
            else
                agent-browser install
            fi

            print_success "Chromium download complete"
        fi

        return 0
    else
        print_error "agent-browser installation failed"
        print_info "Manual install: npm install -g agent-browser"
        return 1
    fi
}

# Initialize config
init_config() {
    print_info "Initializing configuration..."

    if command -v markitai >/dev/null 2>&1; then
        if markitai config init 2>/dev/null; then
            print_success "Configuration initialized"
        fi
    fi
}

# Print completion message
print_completion() {
    printf "\n"
    printf "${GREEN}✓${NC} ${BOLD}Setup complete!${NC}\n"
    printf "\n"
    printf "  ${BOLD}Get started:${NC}\n"
    printf "    ${YELLOW}markitai --help${NC}\n"
    printf "\n"
}

# Main function
main() {
    print_header "Markitai Setup Wizard"

    # Detect Python
    if ! detect_python; then
        exit 1
    fi

    # Detect/install UV
    detect_uv

    # Install markitai
    if ! install_markitai; then
        exit 1
    fi

    # Optional components
    install_optional

    # Initialize config
    init_config

    # Complete
    print_completion
}

# Run main function
main
