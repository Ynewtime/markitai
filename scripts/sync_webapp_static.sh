#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP="$ROOT/webapp"
DIST="$WEBAPP/dist"
TARGET="$ROOT/packages/markitai/src/markitai/serve/static"
MODE="${1:---sync}"

if command -v pnpm >/dev/null 2>&1; then
  PNPM=(pnpm)
else
  PNPM_VERSION="$(
    WEBAPP_PACKAGE_JSON="$WEBAPP/package.json" \
      node -p "require(process.env.WEBAPP_PACKAGE_JSON).packageManager.split('@').pop()"
  )"
  PNPM=(npx --yes "pnpm@$PNPM_VERSION")
fi

"${PNPM[@]}" --dir "$WEBAPP" build

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
