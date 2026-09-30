#!/bin/sh
# Regenerate the Word 97 fixtures: authored HTML saved by macOS textutil, the
# exporter TextEdit uses, with network access denied.
#
# textedit-word97.doc: a short document. The exporter declares a mini stream
#   no stream uses and chains its two MiniFAT entries to sector 0, which
#   strict OLE readers reject ("pointed to twice").
# textedit-word97-long.doc: 205 paragraphs, sized so the exporter writes one
#   FAT sector too few: the FAT sector, the directory and the mini-stream
#   sectors all lie beyond the entries the FAT has ("next_id is invalid").
set -eu
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
cat > "$work/short.html" <<'HTML'
<!doctype html><html><head><meta charset="utf-8"><title>Compound repair</title></head>
<body><h1>Compound repair</h1>
<p>This paragraph was written by the Word 97 exporter of the operating system.</p>
<ul><li>First listed point</li><li>Second listed point</li></ul>
<p>The closing paragraph ends the document.</p></body></html>
HTML
{
  printf '%s\n' '<!doctype html><html><head><meta charset="utf-8"><title>Long compound repair</title></head><body><h1>Long compound repair</h1>'
  i=1
  while [ $i -le 205 ]; do
    printf '<p>Paragraph %s of the long document keeps the Word 97 exporter writing regular sectors until its allocation table is full.</p>\n' $i
    i=$((i + 1))
  done
  printf '%s\n' '<p>The closing paragraph ends the long document.</p></body></html>'
} > "$work/long.html"
for pair in "short.html textedit-word97.doc" "long.html textedit-word97-long.doc"; do
  set -- $pair
  /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' \
    /usr/bin/textutil -convert doc "$work/$1" -output "$here/$2"
done
rm -rf "$work"
