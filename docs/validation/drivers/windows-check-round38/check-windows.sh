#!/bin/sh
# 在 macOS 上对整个 workspace 做 x86_64-pc-windows-msvc 类型检查（cargo check/clippy，不链接）。
# 前提：rustup target add x86_64-pc-windows-msvc；依赖已在本地缓存（离线）。
# ring 的 C 源码用 Apple clang 按 Windows 目标编译，缺失的 C 运行时头文件用最小声明替身；
# 归档用 lib-ar.sh 把 lib.exe 风格参数转成 BSD ar；libsqlite3-sys 改走链接模式（不编译内置 SQLite）。
# 这些产物只用于类型检查，从不链接或运行。
set -eu
root=$(cd "$(dirname "$0")/../../../.." && pwd)
work=${WORK:-$root/.local/windows-check}
stub=$work/crt-stub
mkdir -p "$stub" "$work/sqlite-placeholder"
printf '#pragma once\n#define assert(x) ((void)0)\n' > "$stub/assert.h"
printf '#pragma once\n#include <stddef.h>\nvoid *memcpy(void *, const void *, size_t);\nvoid *memset(void *, int, size_t);\nvoid *memmove(void *, const void *, size_t);\nint memcmp(const void *, const void *, size_t);\nsize_t strlen(const char *);\n' > "$stub/string.h"
printf '#pragma once\n#include <stddef.h>\nvoid abort(void);\nvoid *malloc(size_t);\nvoid free(void *);\nunsigned short _byteswap_ushort(unsigned short);\nunsigned long _byteswap_ulong(unsigned long);\nunsigned long long _byteswap_uint64(unsigned long long);\n' > "$stub/stdlib.h"
for header in stdio.h ctype.h time.h malloc.h windows.h; do printf '#pragma once\n#include <stddef.h>\n' > "$stub/$header"; done
export CARGO_NET_OFFLINE=true LIBSQLITE3_SYS_USE_PKG_CONFIG=1
export SQLITE3_LIB_DIR="$work/sqlite-placeholder" SQLITE3_INCLUDE_DIR="$work/sqlite-placeholder"
export CC_x86_64_pc_windows_msvc=clang AR_x86_64_pc_windows_msvc="$(cd "$(dirname "$0")" && pwd)/lib-ar.sh"
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc -ffreestanding -Wno-ignored-pragma-intrinsic -I$stub"
cd "$root"
command=${1:-check}
[ $# -gt 0 ] && shift
# 例：check-windows.sh clippy -- -D warnings
cargo "$command" --workspace --all-targets --target x86_64-pc-windows-msvc --target-dir "$work/target" "$@"
