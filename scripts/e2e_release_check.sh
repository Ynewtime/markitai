#!/usr/bin/env bash
#
# End-to-end release check — the product, not the code.
#
# The test suite runs against the source tree with the developer's own
# environment around it. This runs against the artifact a user installs, on a
# machine that has never seen markitai, and drives only the public CLI. That
# gap is where the interesting failures live: a flag that silently does
# nothing without a config file, a resume that cannot resume, an install hint
# naming a command that does not work — none of which a unit test sees.
#
# It uses a real LLM and therefore costs real money (a few cents at the
# default model and batch size).
#
#   scripts/e2e_release_check.sh
#
# Configuration — every variable can be overridden from the environment:
#
#   CLEANUP_ON_SUCCESS=1   remove WORKDIR when every check passed
#                          (default 0: artifacts are kept for inspection,
#                          which is the point of running this by hand)
#   WORKDIR=/tmp/...       where the fake home, the wheel and every output
#                          go (default /tmp/markitai-e2e)
#   ENV_FILE=~/.markitai/.env    where provider keys are read from
#   E2E_MODEL=provider/model     pinned model for the deterministic checks
#   BATCH_DOCS=40          documents generated for the batch/interrupt checks
#   HTTP_PORT=8899         port for the local page used by the URL check
#   INTERRUPT_AFTER=6      seconds to let the batch run before interrupting it
#                          (the step shortens the state flush interval so this
#                          does not have to outlast the 10s default)
#   SKIP_INTERRUPT=1       skip the interrupt/resume step entirely
#
# Exit status: 0 when every check passed, 1 otherwise.

set -uo pipefail

CLEANUP_ON_SUCCESS=${CLEANUP_ON_SUCCESS:-0}
WORKDIR=${WORKDIR:-/tmp/markitai-e2e}
ENV_FILE=${ENV_FILE:-$HOME/.markitai/.env}
E2E_MODEL=${E2E_MODEL:-gemini/gemini-flash-lite-latest}
BATCH_DOCS=${BATCH_DOCS:-40}
HTTP_PORT=${HTTP_PORT:-8899}
SKIP_INTERRUPT=${SKIP_INTERRUPT:-0}
INTERRUPT_AFTER=${INTERRUPT_AFTER:-6}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REAL_HOME=$HOME
PASS=0
FAIL=0
SERVER_PID=""

if [ -t 1 ]; then
  B=$(printf '\033[1m'); G=$(printf '\033[32m'); R=$(printf '\033[31m')
  Y=$(printf '\033[33m'); D=$(printf '\033[2m'); N=$(printf '\033[0m')
else
  B=""; G=""; R=""; Y=""; D=""; N=""
fi

step()  { printf '\n%s── %s %s\n' "$B" "$*" "$N"; }
ok()    { PASS=$((PASS + 1)); printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
bad()   { FAIL=$((FAIL + 1)); printf '  %s✗%s %s\n' "$R" "$N" "$*"; }
skip()  { printf '  %s—%s %s\n' "$Y" "$N" "$*"; }
note()  { printf '  %s%s%s\n' "$D" "$*" "$N"; }

# check <description> <condition-command...>
check() { if "${@:2}"; then ok "$1"; else bad "$1"; fi; }

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  return 0
}
trap cleanup EXIT

# ── Setup ────────────────────────────────────────────────────────────────────
step "Setup"

if [ ! -f "$ENV_FILE" ]; then
  printf '%sNo provider keys at %s.%s\n' "$R" "$ENV_FILE" "$N"
  printf 'Set ENV_FILE=/path/to/.env, or export a key before running.\n'
  exit 1
fi
# Read the keys while $HOME still points at the real one.
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
note "keys read from $ENV_FILE"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"/{home,work,site}

(cd "$REPO_ROOT" && uv build --package markitai -o "$WORKDIR/dist") >/dev/null 2>&1 \
  || { printf '%swheel build failed%s\n' "$R" "$N"; exit 1; }
WHEEL=$(ls "$WORKDIR"/dist/markitai-*-py3-none-any.whl | head -1)
note "built $(basename "$WHEEL")"

# A machine that has never seen markitai: empty home, isolated tool dir.
# Exported here only — the caller's shell is untouched.
export HOME="$WORKDIR/home"
export UV_TOOL_DIR="$WORKDIR/home/.uvtools"
export UV_TOOL_BIN_DIR="$WORKDIR/home/bin"
export PATH="$WORKDIR/home/bin:$PATH"
uv tool install "$WHEEL" >/dev/null 2>&1 \
  || { printf '%stool install failed%s\n' "$R" "$N"; exit 1; }
note "installed into a fresh HOME ($HOME)"

cd "$WORKDIR/work" || exit 1
cp "$REPO_ROOT/packages/markitai/tests/fixtures/sample.docx" . 2>/dev/null
cp "$REPO_ROOT/packages/markitai/tests/fixtures/sample.pdf" . 2>/dev/null

# ── 1. First screen ──────────────────────────────────────────────────────────
step "1. First screen — what a new user reads before anything else"
markitai --help >help.txt 2>&1
FIRST_PANEL=$(grep -m1 '^╭─' help.txt | sed 's/[╭─ ]*//; s/ *─*╮*$//')
note "first panel: $FIRST_PANEL"
check "help opens on a curated panel, not ungrouped options" \
  test "$FIRST_PANEL" != "Options"
check "help promises nothing the CLI cannot do (no video conversion)" \
  test "$(grep -ci youtube help.txt)" -eq 0

# ── 2. doctor on a bare machine ──────────────────────────────────────────────
step "2. doctor — the first command the docs tell you to run"
markitai doctor >doctor.txt 2>&1
DOCTOR_RC=$?
check "doctor exits 0 with no config and no extras" test "$DOCTOR_RC" -eq 0
check "every repair hint is one runnable install command" \
  test "$(grep -cE 'pip install "markitai|uv add ' doctor.txt)" -eq 0
note "$(grep -c '•' doctor.txt) hint line(s); read doctor.txt and try one by hand"

# ── 3. Zero-config conversion ────────────────────────────────────────────────
step "3. Convert with no configuration at all"
markitai sample.docx >stdout.md 2>stdout.err
check "stdout is markdown a pipe can consume" \
  test "$(head -c 3 stdout.md)" = "---"
check "converting without a config file is not warned about" \
  test "$(grep -ci 'no config file found' stdout.err)" -eq 0
markitai sample.pdf -o out/ >/dev/null 2>&1
check "writes <name>.md next to its assets" test -f out/sample.pdf.md

# ── 4. LLM: the documented quick path, with a real model ─────────────────────
step "4. LLM enhancement — a real call, billed to your key"
note "auto-detection route: a provider key in the environment, nothing else"
markitai sample.docx -o llm_auto/ --llm >llm_auto.log 2>&1
check "auto-detected provider produced an enhanced file" \
  test -f llm_auto/sample.docx.llm.md
check "the model actually ran (frontmatter carries generated metadata)" \
  grep -q '^description:' llm_auto/sample.docx.llm.md

note "pinned route: MODEL=$E2E_MODEL"
MODEL="$E2E_MODEL" markitai sample.pdf -o llm_pinned/ --llm >llm_pinned.log 2>&1
check "MODEL env var is honoured" grep -q '^description:' llm_pinned/sample.pdf.llm.md
check "--llm alone leaves images alone (alt text is --alt's job)" \
  grep -q '!\[\](' llm_pinned/sample.pdf.llm.md

note "vision route: --alt --desc against a PDF with embedded images"
MODEL="$E2E_MODEL" markitai sample.pdf -o vision/ --llm --alt --desc \
  >vision.log 2>&1
check "--alt writes alt text into the image references" \
  grep -qE '!\[[^]]+\]\(\.markitai/assets/' vision/sample.pdf.llm.md
check "--desc writes the descriptions sidecar" \
  test -s vision/.markitai/assets/images.json
note "$(grep -oE '!\[[^]]{0,60}' vision/sample.pdf.llm.md | head -1)]"
note "cost so far is printed by the batch runs below; single files do not total it"

# ── 5. URL ───────────────────────────────────────────────────────────────────
step "5. URL conversion — main content only"
# Long enough to look like a real article: markitai's quality gate reads a
# two-sentence page as an empty shell and escalates to browser rendering,
# which is the correct call on the web and the wrong one for a fixture.
cat >"$WORKDIR/site/index.html" <<'HTML'
<!doctype html><html><head><title>Quarterly Report</title>
<meta name="author" content="Ops Team"></head><body>
<nav>Home About Contact Careers Press</nav>
<article>
<h1>Quarterly Report</h1>
<p>Revenue grew 14% quarter over quarter, driven by the new pipeline and by a
steadier renewal rate among mid-market accounts. Gross margin held flat while
headcount grew, which is the outcome the plan called for.</p>
<h2>Regional detail</h2>
<p>Region B carried the quarter. Its growth came almost entirely from expansion
inside existing accounts rather than new logos, so the pipeline metrics below
understate how much of the number was already contracted at the start of the
period.</p>
<ul><li>Region A: +9%, in line with plan</li>
<li>Region B: +22%, ahead of plan on expansion</li>
<li>Region C: +3%, behind plan on a delayed launch</li></ul>
<table><tr><th>Region</th><th>Growth</th><th>Plan</th></tr>
<tr><td>A</td><td>9%</td><td>9%</td></tr>
<tr><td>B</td><td>22%</td><td>15%</td></tr></table>
<h2>What we are watching</h2>
<p>Two risks carry into next quarter. The delayed launch in Region C moves
roughly a third of its pipeline into the following period, and the renewal
cohort concentrates in the last three weeks of the quarter, which leaves very
little room to recover a miss.</p>
</article>
<footer>© 2026 Example Corp — all rights reserved</footer></body></html>
HTML
# exec, so $! is the server itself. Without it $! names the subshell, the
# server survives as its orphan, and the *next* run of this script finds the
# port held by the previous one — serving a directory that has since been
# deleted, which reads as "markitai cannot fetch a local page".
(cd "$WORKDIR/site" && exec python3 -m http.server "$HTTP_PORT" >/dev/null 2>&1) &
SERVER_PID=$!

# Confirm the page being served is ours before believing anything the check
# says: an unrelated process already holding the port answers happily, and a
# check that quietly grades someone else's server is worse than no check.
SERVED=""
for _ in 1 2 3 4 5; do
  sleep 1
  SERVED=$(curl -fsS --max-time 2 "http://127.0.0.1:$HTTP_PORT/" 2>/dev/null)
  case "$SERVED" in *"Quarterly Report"*) break ;; esac
done

case "$SERVED" in
  *"Quarterly Report"*) : ;;
  *)
    bad "port $HTTP_PORT is not serving this check's page — another process is \
holding it, or python3 -m http.server did not start (set HTTP_PORT=<free port>)"
    SERVED=""
    ;;
esac

# A local page keeps the check honest on machines whose VPN or DNS rewrites
# public addresses — markitai correctly refuses those as non-public.
URL_MD=""
if [ -n "$SERVED" ]; then
  markitai "http://127.0.0.1:$HTTP_PORT/" -o url/ >url.log 2>&1
  URL_MD=$(ls url/*.md 2>/dev/null | head -1)
fi
if [ -n "$URL_MD" ]; then
  ok "fetched and converted a live page"
  check "page chrome is stripped (no nav, no footer)" \
    test "$(grep -cE 'Home About Contact|Example Corp' "$URL_MD")" -eq 0
  check "metadata lands in frontmatter" grep -q '^author: Ops Team' "$URL_MD"
  check "tables survive" grep -q '| Region' "$URL_MD"
elif [ -n "$SERVED" ]; then
  bad "URL conversion produced no markdown (see url.log)"
fi
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""

# ── 6. Batch ─────────────────────────────────────────────────────────────────
step "6. Batch — $BATCH_DOCS documents with LLM enhancement"
mkdir -p docs
i=1
while [ "$i" -le "$BATCH_DOCS" ]; do
  {
    echo "# Doc $i"
    echo
    j=1
    while [ "$j" -le 40 ]; do
      echo "Paragraph $j of document $i, with   messy   spacing to clean up."
      echo
      j=$((j + 1))
    done
  } >"docs/doc$i.md"
  i=$((i + 1))
done
MODEL="$E2E_MODEL" markitai docs/ -o batch/ --llm --no-cache >batch.log 2>&1
DONE=$(ls batch/*.llm.md 2>/dev/null | wc -l | tr -d ' ')
check "every document was enhanced ($DONE/$BATCH_DOCS)" test "$DONE" -eq "$BATCH_DOCS"
check "the run reports what it spent" grep -qE '\$[0-9]' batch.log
note "$(grep -oE '✓ Done:.*' batch.log | head -1)"

# ── 7. Interrupt and resume ──────────────────────────────────────────────────
step "7. Interrupt and resume"
if [ "$SKIP_INTERRUPT" = "1" ]; then
  skip "SKIP_INTERRUPT=1"
else
  # The script sends the interrupt rather than asking you to press Ctrl-C.
  # Ctrl-C goes to the whole foreground process group, so it would take this
  # script down with markitai and the resume half would never run. What is
  # being checked is markitai's behaviour on SIGINT, which is the same either
  # way.
  #
  # The launcher exists because a shell that is not interactive starts its
  # background children with SIGINT already ignored, and Python keeps an
  # inherited SIG_IGN — signal markitai without this and it runs to
  # completion, which reads convincingly like "Ctrl-C does nothing".
  cat >"$WORKDIR/interrupt_launcher.py" <<'PYEOF'
import os
import signal
import sys

signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
PYEOF

  # Completions reach the state file on an interval (10s by default), so a
  # batch that finishes in under ~20s can only ever be interrupted inside
  # that window, with nothing recorded to resume from. Shorten the interval
  # for these two runs instead of generating enough documents to outlast it:
  # same machinery, a fraction of the LLM spend. Both runs pass the same
  # override so they agree on the state file.
  FLUSH_OVERRIDE='{"batch":{"state_flush_interval_seconds":2}}'

  MODEL="$E2E_MODEL" python3 "$WORKDIR/interrupt_launcher.py" \
    markitai docs/ -o resume/ --llm --no-cache \
    --config-json "$FLUSH_OVERRIDE" >interrupt.log 2>&1 &
  RUN_PID=$!
  sleep "$INTERRUPT_AFTER"
  kill -INT "$RUN_PID" 2>/dev/null
  wait "$RUN_PID" 2>/dev/null
  RUN_RC=$?

  PARTIAL=$(ls resume/*.llm.md 2>/dev/null | wc -l | tr -d ' ')
  if [ "$PARTIAL" -ge "$BATCH_DOCS" ]; then
    skip "the batch finished inside ${INTERRUPT_AFTER}s — raise BATCH_DOCS or lower INTERRUPT_AFTER"
  elif [ "$PARTIAL" -eq 0 ]; then
    skip "nothing had finished at ${INTERRUPT_AFTER}s — raise INTERRUPT_AFTER"
  else
    note "interrupted with $PARTIAL/$BATCH_DOCS written"
    check "an interrupt stops the run" test "$RUN_RC" -ne 0
    check "an interrupted batch says how to continue" \
      grep -q -- '--resume' interrupt.log
    check "resumable state survived the interrupt" \
      test -n "$(ls resume/.markitai/states/*.state.json 2>/dev/null)"

    MODEL="$E2E_MODEL" markitai docs/ -o resume/ --llm --no-cache --resume \
      --config-json "$FLUSH_OVERRIDE" >resume.log 2>&1
    check "resume reads the previous run's state" grep -q 'Resuming batch' resume.log
    note "$(grep -oE 'Resuming batch.*' resume.log | head -1)"

    # The line alone is not the point — a resume that reports 0 completed has
    # restarted, and the user pays for the whole batch again. Completions are
    # flushed on an interval (batch.state_flush_interval_seconds, 10s by
    # default), so an interrupt inside that window genuinely has nothing to
    # skip; say which of the two happened instead of scoring it as a pass.
    CARRIED=$(grep -oE 'Resuming batch: [0-9]+' resume.log | grep -oE '[0-9]+' | head -1)
    CARRIED=${CARRIED:-0}
    if [ "$CARRIED" -gt 0 ]; then
      ok "resume skipped $CARRIED document(s) instead of redoing them"
    else
      skip "resume carried nothing over: nothing had been flushed when the interrupt landed — raise INTERRUPT_AFTER and re-run"
    fi

    check "resume completes the batch" \
      test "$(ls resume/*.llm.md 2>/dev/null | wc -l | tr -d ' ')" -eq "$BATCH_DOCS"
  fi
fi

# ── 8. Cache ─────────────────────────────────────────────────────────────────
step "8. Cache — the same work must not be paid for twice"
MODEL="$E2E_MODEL" markitai docs/ -o cache1/ --llm >cache1.log 2>&1
START=$(date +%s)
MODEL="$E2E_MODEL" markitai docs/ -o cache2/ --llm >cache2.log 2>&1
ELAPSED=$(( $(date +%s) - START ))
check "a repeat run is served from cache" grep -qE 'Cache: [0-9]+' cache2.log
check "and is fast (${ELAPSED}s)" test "$ELAPSED" -le 10

# ── 9. Failure paths ─────────────────────────────────────────────────────────
step "9. Failures — the messages you meet on a bad day"
markitai /definitely/not/here.txt >missing.log 2>&1
check "a missing file exits non-zero" test $? -ne 0
# An image, not the PDF fixture: a born-digital PDF has a text layer, so --ocr
# on it is a no-op that succeeds and never reaches the missing backend.
python3 - <<'PNG'
import base64, pathlib
pathlib.Path("probe.png").write_bytes(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAFklEQVR4nGP8//8/AzGAiShVowZS"
    "z0AAQFgBBSBqfMcAAAAASUVORK5CYII="))
PNG
markitai probe.png --ocr -o ocr/ >ocr.log 2>&1
if grep -qE '(uv tool|pipx|pip) install' ocr.log; then
  ok "a missing extra fails loudly and names the command that fixes it"
  note "$(grep -oE '(uv tool|pipx|pip) install [^ ]*[^ ]*( --force)?' ocr.log | head -1)"
elif grep -qi 'rapidocr' ocr.log; then
  skip "the ocr extra is installed in this environment; nothing to report"
else
  bad "--ocr without the backend did not explain itself (see ocr.log)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
step "Summary"
printf '  %s%d passed%s, %s%d failed%s\n' "$G" "$PASS" "$N" \
  "$([ "$FAIL" -gt 0 ] && printf '%s' "$R")" "$FAIL" "$N"

if [ "$FAIL" -eq 0 ] && [ "$CLEANUP_ON_SUCCESS" = "1" ]; then
  cd "$REAL_HOME" || cd /
  rm -rf "$WORKDIR"
  printf '  %scleaned up %s (CLEANUP_ON_SUCCESS=1)%s\n' "$D" "$WORKDIR" "$N"
else
  printf '  artifacts kept in %s%s%s\n' "$B" "$WORKDIR" "$N"
  printf '  %sread work/*.log and work/*/ to judge quality by eye;%s\n' "$D" "$N"
  printf '  %sre-run with CLEANUP_ON_SUCCESS=1 to remove them on a clean pass%s\n' "$D" "$N"
fi

[ "$FAIL" -eq 0 ]
