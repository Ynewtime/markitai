#!/usr/bin/env bash
# scripts/sync_defuddle_fixtures.sh
# Copy test fixtures from local defuddle repo at a pinned commit.
# Usage: ./scripts/sync_defuddle_fixtures.sh /path/to/defuddle

set -euo pipefail

DEFUDDLE_DIR="${1:?Usage: $0 /path/to/defuddle}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/packages/markitai/tests/defuddle_fixtures"

if [[ ! -d "$DEFUDDLE_DIR/tests/fixtures" ]]; then
    echo "Error: $DEFUDDLE_DIR/tests/fixtures not found" >&2
    exit 1
fi

# Ensure destination exists before writing anything
mkdir -p "$DEST_DIR/fixtures" "$DEST_DIR/expected"

# Record version
COMMIT=$(git -C "$DEFUDDLE_DIR" rev-parse HEAD)
echo "defuddle commit: $COMMIT" > "$DEST_DIR/VERSION"
echo "synced at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DEST_DIR/VERSION"

# Sync fixtures (clean then copy)
rm -rf "$DEST_DIR/fixtures/"* "$DEST_DIR/expected/"*
cp "$DEFUDDLE_DIR"/tests/fixtures/*.html "$DEST_DIR/fixtures/"
cp "$DEFUDDLE_DIR"/tests/expected/*.md "$DEST_DIR/expected/"

echo "Synced $(ls "$DEST_DIR/fixtures/" | wc -l) fixtures, $(ls "$DEST_DIR/expected/" | wc -l) expected files"
echo "From defuddle commit: $COMMIT"
