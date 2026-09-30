#!/bin/sh
# MSVC lib.exe 风格参数 -> BSD ar（仅用于 cargo check，产物不链接）
out=""; files=""
for a in "$@"; do
  case "$a" in
    -out:*|/OUT:*) out="${a#*:}" ;;
    -nologo|/NOLOGO) ;;
    *) files="$files $a" ;;
  esac
done
rm -f "$out"
exec /usr/bin/ar crs "$out" $files
