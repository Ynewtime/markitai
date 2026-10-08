#!/usr/bin/env bash
# scripts/sync_defuddle_fixtures.sh
# Copy test fixtures from a local defuddle checkout and pin its HEAD commit.
# Check out the defuddle commit you want first: the pin recorded in VERSION
# and PORT_MANIFEST.md is whatever HEAD is when this runs.
# Usage: ./scripts/sync_defuddle_fixtures.sh /path/to/defuddle

set -euo pipefail

DEFUDDLE_DIR="${1:?Usage: $0 /path/to/defuddle}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/packages/markitai/tests/defuddle_fixtures"
MANIFEST="$REPO_ROOT/packages/markitai/src/markitai/webextract/PORT_MANIFEST.md"

if [[ ! -d "$DEFUDDLE_DIR/tests/fixtures" ]]; then
    echo "Error: $DEFUDDLE_DIR/tests/fixtures not found" >&2
    exit 1
fi

# Uncommitted fixture edits would not match the commit we are about to pin
if [[ -n "$(git -C "$DEFUDDLE_DIR" status --porcelain -- tests/fixtures tests/expected)" ]]; then
    echo "Error: $DEFUDDLE_DIR has uncommitted changes under tests/; commit or stash them first" >&2
    exit 1
fi
COMMIT=$(git -C "$DEFUDDLE_DIR" rev-parse HEAD)

# Stage the copy next to the destination, so a failed copy leaves the
# current corpus and its pins untouched
mkdir -p "$DEST_DIR"
STAGE="$(mktemp -d "$DEST_DIR/.sync.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/fixtures" "$STAGE/expected"
cp "$DEFUDDLE_DIR"/tests/fixtures/*.html "$STAGE/fixtures/"
cp "$DEFUDDLE_DIR"/tests/expected/*.md "$STAGE/expected/"

# Swap the staged corpus in
rm -rf "$DEST_DIR/fixtures" "$DEST_DIR/expected"
mv "$STAGE/fixtures" "$STAGE/expected" "$DEST_DIR/"

# Pin only once the corpus is complete. The port manifest pin moves in
# lockstep with VERSION (tests/unit/webextract/test_port_manifest.py
# enforces they match).
echo "defuddle commit: $COMMIT" > "$DEST_DIR/VERSION"
echo "synced at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DEST_DIR/VERSION"
sed "s/^Pinned upstream commit: .*/Pinned upstream commit: \`$COMMIT\`/" \
    "$MANIFEST" > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"

echo "Synced $(ls "$DEST_DIR/fixtures/" | wc -l) fixtures, $(ls "$DEST_DIR/expected/" | wc -l) expected files"
echo "From defuddle commit: $COMMIT"
