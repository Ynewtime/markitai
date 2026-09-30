#!/bin/sh
# Print borderless-table.html to a tagged PDF with headless Chrome, as the local
# PDF corpus is printed: a private profile, no network, no header or footer.
# Chrome can stay running after printing, so it is stopped once the PDF is
# written (or after 60 seconds).
set -eu
cd "$(dirname "$0")"
profile=$(mktemp -d)
rm -f borderless-table.pdf
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --no-first-run --no-default-browser-check --disable-background-networking \
  --disable-component-update --disable-sync '--host-resolver-rules=MAP * ~NOTFOUND' \
  "--user-data-dir=$profile" --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=2000 "--print-to-pdf=$PWD/borderless-table.pdf" \
  "file://$PWD/borderless-table.html" &
chrome=$!
for _ in $(seq 60); do
  [ -s borderless-table.pdf ] && break
  sleep 1
done
sleep 1
kill "$chrome" 2>/dev/null || true
wait "$chrome" 2>/dev/null || true
rm -rf "$profile"
[ -s borderless-table.pdf ]
