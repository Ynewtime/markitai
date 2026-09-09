#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP="$ROOT/webapp"
DIST="$WEBAPP/dist"
TARGET="$ROOT/packages/markitai/src/markitai/serve/static"
MODE="${1:---sync}"

bun install --cwd "$WEBAPP" --frozen-lockfile
bun run --cwd "$WEBAPP" build

case "$MODE" in
  --sync)
    rm -rf "$TARGET"
    mkdir -p "$TARGET"
    cp -R "$DIST"/. "$TARGET"/
    ;;
  --check)
    if ! diff -qr "$DIST" "$TARGET"; then
      echo "Packaged webapp is stale. Run scripts/sync_webapp_static.sh." >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [--sync|--check]" >&2
    exit 2
    ;;
esac
