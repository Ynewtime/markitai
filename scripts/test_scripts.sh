#!/bin/bash
# 脚本测试套件
# 测试 setup 脚本的核心功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_PASSED=0
TESTS_FAILED=0

# 测试辅助函数
assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="$3"
    if [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}✓${NC} $msg"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $msg"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

assert_true() {
    local condition="$1"
    local msg="$2"
    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $msg"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $msg"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local msg="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo -e "${GREEN}✓${NC} $msg"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $msg"
        echo "  Expected to contain: $needle"
        echo "  Actual: $haystack"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# ============================================================
# 测试 lib.sh
# ============================================================

test_lib_sh_syntax() {
    echo -e "\n${YELLOW}=== 测试 lib.sh 语法 ===${NC}"
    bash -n "$SCRIPT_DIR/lib.sh"
    assert_eq "0" "$?" "lib.sh 语法正确"
}

test_lib_sh_loads() {
    echo -e "\n${YELLOW}=== 测试 lib.sh 加载 ===${NC}"
    (
        . "$SCRIPT_DIR/lib.sh"
        # 检查关键函数是否定义
        assert_true "type lib_detect_python >/dev/null 2>&1" "lib_detect_python 函数已定义"
        assert_true "type lib_install_uv >/dev/null 2>&1" "lib_install_uv 函数已定义"
        assert_true "type print_success >/dev/null 2>&1" "print_success 函数已定义"
        assert_true "type print_error >/dev/null 2>&1" "print_error 函数已定义"
        assert_true "type ask_yes_no >/dev/null 2>&1" "ask_yes_no 函数已定义"
    )
}

test_lib_detect_python_uses_uv() {
    echo -e "\n${YELLOW}=== 测试 lib_detect_python 使用 uv ===${NC}"
    (
        . "$SCRIPT_DIR/lib.sh"
        # 检查函数实现中包含 uv python find
        local func_body
        func_body=$(type lib_detect_python)
        assert_contains "$func_body" "uv python find" "lib_detect_python 使用 uv python find"
        assert_contains "$func_body" "uv python install" "lib_detect_python 使用 uv python install"
    )
}

# ============================================================
# 测试 setup-dev-zh.sh
# ============================================================

test_setup_dev_zh_syntax() {
    echo -e "\n${YELLOW}=== 测试 setup-dev-zh.sh 语法 ===${NC}"
    bash -n "$SCRIPT_DIR/setup-dev-zh.sh"
    assert_eq "0" "$?" "setup-dev-zh.sh 语法正确"
}

test_setup_dev_zh_uv_before_python() {
    echo -e "\n${YELLOW}=== 测试 setup-dev-zh.sh 流程顺序 ===${NC}"
    local content
    content=$(cat "$SCRIPT_DIR/setup-dev-zh.sh")

    # 找到 UV 和 Python 检测的行号
    local uv_line python_line
    uv_line=$(grep -n "检测 UV 包管理器" "$SCRIPT_DIR/setup-dev-zh.sh" | head -1 | cut -d: -f1)
    python_line=$(grep -n "检测 Python" "$SCRIPT_DIR/setup-dev-zh.sh" | head -1 | cut -d: -f1)

    assert_true "[ $uv_line -lt $python_line ]" "UV 检测在 Python 检测之前 (行 $uv_line < $python_line)"
}

test_setup_dev_zh_detect_python_uses_uv() {
    echo -e "\n${YELLOW}=== 测试 zh_detect_python 使用 uv ===${NC}"
    local content
    content=$(cat "$SCRIPT_DIR/setup-dev-zh.sh")

    assert_contains "$content" "uv python find 3.13" "zh_detect_python 使用 uv python find 3.13"
    assert_contains "$content" "uv python install 3.13" "zh_detect_python 使用 uv python install 3.13"
}

# ============================================================
# 测试 setup-zh.sh
# ============================================================

test_setup_zh_syntax() {
    echo -e "\n${YELLOW}=== 测试 setup-zh.sh 语法 ===${NC}"
    bash -n "$SCRIPT_DIR/setup-zh.sh"
    assert_eq "0" "$?" "setup-zh.sh 语法正确"
}

test_setup_zh_uv_before_python() {
    echo -e "\n${YELLOW}=== 测试 setup-zh.sh 流程顺序 ===${NC}"
    local uv_line python_line
    uv_line=$(grep -n "检测 UV 包管理器" "$SCRIPT_DIR/setup-zh.sh" | head -1 | cut -d: -f1)
    python_line=$(grep -n "检测 Python" "$SCRIPT_DIR/setup-zh.sh" | head -1 | cut -d: -f1)

    assert_true "[ $uv_line -lt $python_line ]" "UV 检测在 Python 检测之前 (行 $uv_line < $python_line)"
}

# ============================================================
# 测试 setup-dev.sh
# ============================================================

test_setup_dev_syntax() {
    echo -e "\n${YELLOW}=== 测试 setup-dev.sh 语法 ===${NC}"
    bash -n "$SCRIPT_DIR/setup-dev.sh"
    assert_eq "0" "$?" "setup-dev.sh 语法正确"
}

test_setup_dev_uv_before_python() {
    echo -e "\n${YELLOW}=== 测试 setup-dev.sh 流程顺序 ===${NC}"
    local uv_line python_line
    uv_line=$(grep -n "Detecting UV package manager" "$SCRIPT_DIR/setup-dev.sh" | head -1 | cut -d: -f1)
    python_line=$(grep -n "Detecting Python" "$SCRIPT_DIR/setup-dev.sh" | head -1 | cut -d: -f1)

    assert_true "[ $uv_line -lt $python_line ]" "UV 检测在 Python 检测之前 (行 $uv_line < $python_line)"
}

# ============================================================
# 测试 setup.sh
# ============================================================

test_setup_syntax() {
    echo -e "\n${YELLOW}=== 测试 setup.sh 语法 ===${NC}"
    bash -n "$SCRIPT_DIR/setup.sh"
    assert_eq "0" "$?" "setup.sh 语法正确"
}

test_setup_uv_before_python() {
    echo -e "\n${YELLOW}=== 测试 setup.sh 流程顺序 ===${NC}"
    local uv_line python_line
    uv_line=$(grep -n "Detecting UV package manager" "$SCRIPT_DIR/setup.sh" | head -1 | cut -d: -f1)
    python_line=$(grep -n "Detecting Python" "$SCRIPT_DIR/setup.sh" | head -1 | cut -d: -f1)

    assert_true "[ $uv_line -lt $python_line ]" "UV 检测在 Python 检测之前 (行 $uv_line < $python_line)"
}

# ============================================================
# 测试 Arch Linux 支持
# ============================================================

test_arch_linux_playwright_deps() {
    echo -e "\n${YELLOW}=== 测试 Arch Linux Playwright 依赖支持 ===${NC}"

    # 检查所有脚本是否包含 Arch Linux 检测
    for script in lib.sh setup-dev.sh setup-dev-zh.sh setup-zh.sh; do
        local content
        content=$(cat "$SCRIPT_DIR/$script")
        assert_contains "$content" "/etc/arch-release" "$script 包含 Arch Linux 检测"
        assert_contains "$content" "pacman -S" "$script 包含 pacman 安装命令"
    done
}

test_arch_linux_playwright_packages() {
    echo -e "\n${YELLOW}=== 测试 Arch Linux Playwright 包列表 ===${NC}"
    local content
    content=$(cat "$SCRIPT_DIR/setup-dev-zh.sh")

    # 核心依赖必须存在
    assert_contains "$content" "nss" "包含 nss 依赖"
    assert_contains "$content" "nspr" "包含 nspr 依赖"
    assert_contains "$content" "at-spi2-core" "包含 at-spi2-core 依赖"
    assert_contains "$content" "alsa-lib" "包含 alsa-lib 依赖"
    assert_contains "$content" "mesa" "包含 mesa 依赖"
}

# ============================================================
# 集成测试：实际运行脚本
# ============================================================

test_setup_dev_zh_runs() {
    echo -e "\n${YELLOW}=== 测试 setup-dev-zh.sh 实际运行 ===${NC}"
    local output
    output=$(timeout 30 "$SCRIPT_DIR/setup-dev-zh.sh" 2>&1 </dev/null | head -30) || true

    # 检查输出中包含正确的步骤顺序
    assert_contains "$output" "[1/5]" "输出包含步骤 1"
    assert_contains "$output" "UV" "步骤 1 涉及 UV"
    assert_contains "$output" "Python" "输出包含 Python 相关信息"
}

# ============================================================
# 运行所有测试
# ============================================================

main() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Markitai Scripts 测试套件${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # lib.sh 测试
    test_lib_sh_syntax
    test_lib_sh_loads
    test_lib_detect_python_uses_uv

    # setup-dev-zh.sh 测试
    test_setup_dev_zh_syntax
    test_setup_dev_zh_uv_before_python
    test_setup_dev_zh_detect_python_uses_uv

    # setup-zh.sh 测试
    test_setup_zh_syntax
    test_setup_zh_uv_before_python

    # setup-dev.sh 测试
    test_setup_dev_syntax
    test_setup_dev_uv_before_python

    # Arch Linux 支持测试
    test_arch_linux_playwright_deps
    test_arch_linux_playwright_packages

    # setup.sh 测试
    test_setup_syntax
    test_setup_uv_before_python

    # 集成测试
    test_setup_dev_zh_runs

    # 总结
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  测试结果${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "  ${GREEN}通过: $TESTS_PASSED${NC}"
    echo -e "  ${RED}失败: $TESTS_FAILED${NC}"

    if [ $TESTS_FAILED -gt 0 ]; then
        exit 1
    fi
}

main "$@"
