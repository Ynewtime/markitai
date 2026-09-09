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
# What it drives, in order: the first help screen, doctor on a bare machine,
# zero-config conversion, the three LLM routes (key, MODEL, --alt/--desc), a
# local web page, a directory batch with cost reporting, interrupt + resume,
# the cache, the failure messages, every documented input format, presets and
# output profiles, the flags that change a run's shape, the refusals (removed
# aliases, private URLs offered to remote services, Batch API on the wrong
# pool), a .urls list, the three configuration layers plus the cache
# commands, the Python API, then — with the serve/mcp/legacy extras added to
# the same install — doctor again, legacy Office formats, the serve API the
# browser UI talks to, and markitai-mcp over stdio; then, with browser,
# and extra-fetch added too, Chromium via doctor --fix, Playwright
# rendering, URL screenshots and screenshot-only reading, the Cloudflare
# file backend, the remote extraction strategies on
# a public page (skipped with the reason when the resolver is a fake-IP
# VPN), and a real Batch API job through submit, hand-off and collect.
#
# It writes a report you can open — WORKDIR/report.html — with one numbered
# directory per step beside it holding that step's inputs, outputs and logs.
#
# Configuration — every variable can be overridden from the environment:
#
#   CLEANUP_ON_SUCCESS=1   remove WORKDIR when every check passed
#                          (default 0: the artifacts and the report are the
#                          point of running this by hand)
#   WORKDIR=/tmp/...       where the report, the artifacts and the throwaway
#                          home go (default /tmp/markitai-e2e)
#   ENV_FILE=~/.markitai/.env    where provider keys are read from
#   E2E_MODEL=provider/model     pinned model for the deterministic checks
#   E2E_PYTHON=/path/to/python   interpreter for the throwaway install
#                          (default: the repo's own venv interpreter)
#   BATCH_DOCS=40          documents generated for the batch/interrupt steps
#   HTTP_PORT=8899         port for the local page used by the URL steps
#   SERVE_PORT=8900        port for the web workspace step
#   E2E_PUBLIC_URL=https://...   public page for the remote-strategy step
#   E2E_BATCH_MODEL=openai/...   single-model pool for the Batch API step
#   BATCH_WAIT=600         seconds to keep collecting a Batch API job that
#                          outlasted its 120s submit wait
#   PLAYWRIGHT_BROWSERS_PATH=...  where Chromium lives (default: the real
#                          home's Playwright cache, so runs share one download;
#                          point it at an empty dir to test the cold path)
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
SERVE_PORT=${SERVE_PORT:-8900}
E2E_PUBLIC_URL=${E2E_PUBLIC_URL:-https://github.com/Ynewtime/markitai}
E2E_BATCH_MODEL=${E2E_BATCH_MODEL:-openai/gpt-5.6-luna}
BATCH_WAIT=${BATCH_WAIT:-600}
INTERRUPT_AFTER=${INTERRUPT_AFTER:-6}
SKIP_INTERRUPT=${SKIP_INTERRUPT:-0}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REAL_HOME=$HOME
STARTED_EPOCH=$(date +%s)
PASS=0
FAIL=0
SERVER_PID=""
RUN_PID=""
STEP_ID=""

if [ -t 1 ]; then
  B=$(printf '\033[1m'); G=$(printf '\033[32m'); R=$(printf '\033[31m')
  Y=$(printf '\033[33m'); D=$(printf '\033[2m'); N=$(printf '\033[0m')
else
  B=""; G=""; R=""; Y=""; D=""; N=""
fi

# Terminal output and the report are fed by the same calls: a check that
# appears in only one of the two is a check someone will stop trusting.
log() { printf '%s\n' "$(printf '%s\t' "$@" | sed 's/\t$//')" >>"$RESULTS"; }

step() {
  STEP_ID=$1
  printf '\n%s── %s%s\n' "$B" "$2" "$N"
  mkdir -p "$WORKDIR/$1"
  log STEP "$1" "$2" "$3"
}
ok()   { PASS=$((PASS + 1)); printf '  %s✓%s %s\n' "$G" "$N" "$1"; log CHECK "$STEP_ID" ok "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  %s✗%s %s\n' "$R" "$N" "$1"; log CHECK "$STEP_ID" fail "$1"; }
skip() { printf '  %s—%s %s\n' "$Y" "$N" "$1"; log CHECK "$STEP_ID" skip "$1"; }
note() { printf '  %s%s%s\n' "$D" "$1" "$N"; log NOTE "$STEP_ID" "$1"; }
show() { log EVIDENCE "$STEP_ID" "$1" "$2" inline; }   # rendered into the report
file() { log EVIDENCE "$STEP_ID" "$1" "$2"; }          # linked from the report

check() { if "${@:2}"; then ok "$1"; else bad "$1"; fi; }

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  [ -n "$RUN_PID" ] && kill "$RUN_PID" 2>/dev/null
  return 0
}
trap cleanup EXIT

# ── Setup ────────────────────────────────────────────────────────────────────
printf '%s── Setup%s\n' "$B" "$N"

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

# WORKDIR is caller-supplied and gets removed wholesale: only ever remove a
# directory this script created, identified by the marker it writes below.
MARKER="$WORKDIR/_internal/.markitai-e2e"
if [ -e "$WORKDIR" ] && [ ! -f "$MARKER" ]; then
  printf '%s%s exists and was not created by this script — refusing to remove it.%s\n' \
    "$R" "$WORKDIR" "$N"
  printf 'Set WORKDIR=/path/to/a/throwaway/dir, or delete it yourself.\n'
  exit 1
fi
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"/00-inputs "$WORKDIR"/_internal
: >"$MARKER"
RESULTS="$WORKDIR/_internal/results.tsv"
: >"$RESULTS"
printf '  %skeys read from %s%s\n' "$D" "$ENV_FILE" "$N"

(cd "$REPO_ROOT" && uv build --package markitai -o "$WORKDIR/_internal/dist") \
  >"$WORKDIR/_internal/build.log" 2>&1 \
  || { printf '%swheel build failed — see %s/_internal/build.log%s\n' "$R" "$WORKDIR" "$N"; exit 1; }
WHEEL=$(ls "$WORKDIR"/_internal/dist/markitai-*-py3-none-any.whl | head -1)
VERSION=$(basename "$WHEEL" | sed -E 's/^markitai-(.+)-py3-none-any\.whl$/\1/')
printf '  %sbuilt markitai %s%s\n' "$D" "$VERSION" "$N"

# A machine that has never seen markitai: empty home, isolated tool dir.
# Exported here only — the caller's shell is untouched. The uv download
# cache is deliberately kept: isolation is about markitai's config and
# tools, and a cold cache turns this step into a 200MB download that looks
# like a hang.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(uv cache dir 2>/dev/null || printf '%s' "$HOME/.cache/uv")}"
# Pin the interpreter to the one the repo develops against: without it uv
# picks whatever python is first on the machine (here a 3.14 the package
# does not support), and none of the cached cp312 wheels apply.
E2E_PYTHON="${E2E_PYTHON:-$(uv python find --project "$REPO_ROOT")}"
case "$(uname -s)" in
  Darwin) _PW_DEFAULT="$HOME/Library/Caches/ms-playwright" ;;
  *) _PW_DEFAULT="$HOME/.cache/ms-playwright" ;;
esac
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$_PW_DEFAULT}"
export HOME="$WORKDIR/_internal/home"
export UV_TOOL_DIR="$HOME/.uvtools"
export UV_TOOL_BIN_DIR="$HOME/bin"
export PATH="$HOME/bin:$PATH"
mkdir -p "$HOME"
printf '  %sinstalling into an empty home with %s (log: %s/_internal/install.log)%s\n' "$D" "$E2E_PYTHON" "$WORKDIR" "$N"
uv tool install --python "$E2E_PYTHON" "$WHEEL" >"$WORKDIR/_internal/install.log" 2>&1 \
  || { printf '%stool install failed — see %s/_internal/install.log%s\n' "$R" "$WORKDIR" "$N"; exit 1; }
printf '  %sinstalled into an empty home%s\n' "$D" "$N"

cd "$WORKDIR" || exit 1
cp "$REPO_ROOT/packages/markitai/tests/fixtures/sample.docx" 00-inputs/ 2>/dev/null
cp "$REPO_ROOT/packages/markitai/tests/fixtures/sample.pdf" 00-inputs/ 2>/dev/null

# ── 1 ────────────────────────────────────────────────────────────────────────
step 01-first-screen "First screen" \
  "What someone reads in the first ten seconds, before deciding whether to keep going."
markitai --help >01-first-screen/help.txt 2>&1
FIRST_PANEL=$(grep -m1 '^╭─' 01-first-screen/help.txt | sed 's/[╭─ ]*//; s/ *─*╮*$//')
note "first panel shown: $FIRST_PANEL"
check "the help opens on a curated panel, not on ungrouped leftovers" \
  test "$FIRST_PANEL" != "Options"
check "the examples promise nothing the tool cannot do" \
  test "$(grep -ci youtube 01-first-screen/help.txt)" -eq 0
file "full help output" 01-first-screen/help.txt

# ── 2 ────────────────────────────────────────────────────────────────────────
step 02-doctor "Diagnostics on a bare machine" \
  "The first command the docs send you to. Its advice has to be copy-pasteable and correct."
markitai doctor >02-doctor/doctor.txt 2>&1
DOCTOR_RC=$?
check "doctor exits cleanly with no config and no optional extras" \
  test "$DOCTOR_RC" -eq 0
check "every repair hint is a single command that works for this install" \
  test "$(grep -cE 'pip install "markitai|uv add ' 02-doctor/doctor.txt)" -eq 0
show "what a new user is told is missing, and how to fix it" 02-doctor/doctor.txt

# ── 3 ────────────────────────────────────────────────────────────────────────
step 03-zero-config "Conversion with no setup" \
  "The promise on the front page: no API key, no config file, no optional dependency."
markitai 00-inputs/sample.docx >03-zero-config/stdout.md 2>03-zero-config/stderr.txt
check "piping to stdout yields markdown and nothing else" \
  test "$(head -c 3 03-zero-config/stdout.md)" = "---"
check "running without a config file is not treated as a problem" \
  test "$(grep -ci 'no config file found' 03-zero-config/stderr.txt)" -eq 0
markitai 00-inputs/sample.pdf -o 03-zero-config/output/ >03-zero-config/convert.log 2>&1
check "a PDF converts and lands where it was asked to" \
  test -f 03-zero-config/output/sample.pdf.md
show "the markdown a plain conversion produces" 03-zero-config/output/sample.pdf.md

# ── 4 ────────────────────────────────────────────────────────────────────────
step 04-llm-enhancement "LLM enhancement, billed to a real key" \
  "Three routes onto a model, and the line between cleaning text and looking at pictures."
note "route 1 — a provider key in the environment and nothing else"
markitai 00-inputs/sample.docx -o 04-llm-enhancement/auto-detected/ --llm \
  >04-llm-enhancement/auto-detected.log 2>&1
check "a key alone is enough to enable enhancement" \
  test -f 04-llm-enhancement/auto-detected/sample.docx.llm.md
check "the model really ran (frontmatter carries generated metadata)" \
  grep -q '^description:' 04-llm-enhancement/auto-detected/sample.docx.llm.md

note "route 2 — MODEL=$E2E_MODEL pins one model"
MODEL="$E2E_MODEL" markitai 00-inputs/sample.pdf -o 04-llm-enhancement/pinned-model/ \
  --llm >04-llm-enhancement/pinned-model.log 2>&1
check "MODEL is honoured" \
  grep -q '^description:' 04-llm-enhancement/pinned-model/sample.pdf.llm.md
check "--llm on its own leaves images untouched — alt text is --alt's job" \
  grep -q '!\[\](' 04-llm-enhancement/pinned-model/sample.pdf.llm.md

note "route 3 — --alt --desc adds vision analysis of the embedded images"
MODEL="$E2E_MODEL" markitai 00-inputs/sample.pdf -o 04-llm-enhancement/vision-alt-desc/ \
  --llm --alt --desc >04-llm-enhancement/vision.log 2>&1
check "--alt writes alt text into every image reference" \
  grep -qE '!\[[^]]+\]\(\.markitai/assets/' 04-llm-enhancement/vision-alt-desc/sample.pdf.llm.md
check "--desc writes the descriptions sidecar" \
  test -s 04-llm-enhancement/vision-alt-desc/.markitai/assets/images.json
ALT=$(grep -oE '!\[[^]]{1,90}\]' 04-llm-enhancement/vision-alt-desc/sample.pdf.llm.md | head -1)
[ -n "$ALT" ] && note "alt text the model produced: $ALT"
show "enhanced output, with alt text" \
  04-llm-enhancement/vision-alt-desc/sample.pdf.llm.md
file "plain --llm output, for comparison" \
  04-llm-enhancement/pinned-model/sample.pdf.llm.md
file "image descriptions" \
  04-llm-enhancement/vision-alt-desc/.markitai/assets/images.json

# ── 5 ────────────────────────────────────────────────────────────────────────
step 05-url "A web page, reduced to its article" \
  "Fetching is the easy half; the value is in what gets thrown away."
# Long enough to look like a real article: markitai's quality gate reads a
# two-sentence page as an empty shell and escalates to browser rendering,
# which is the correct call on the web and the wrong one for a fixture.
mkdir -p 00-inputs/site
cat >00-inputs/site/index.html <<'HTML'
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
<tr><td>A</td><td>9%</td><td>15%</td></tr>
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
(cd 00-inputs/site && exec python3 -m http.server "$HTTP_PORT" >/dev/null 2>&1) &
SERVER_PID=$!

# Confirm the page being served is ours before believing anything this step
# says: an unrelated process already holding the port answers happily, and a
# check that quietly grades someone else's server is worse than no check.
SERVED=""
for _ in 1 2 3 4 5; do
  sleep 1
  SERVED=$(curl -fsS --max-time 2 "http://127.0.0.1:$HTTP_PORT/" 2>/dev/null)
  case "$SERVED" in *"Quarterly Report"*) break ;; esac
done

URL_MD=""
case "$SERVED" in
  *"Quarterly Report"*)
    # A local page keeps this honest on machines whose VPN or DNS rewrites
    # public addresses — markitai correctly refuses those as non-public.
    markitai "http://127.0.0.1:$HTTP_PORT/" -o 05-url/output/ >05-url/fetch.log 2>&1
    URL_MD=$(ls 05-url/output/*.md 2>/dev/null | head -1)
    ;;
  *)
    bad "port $HTTP_PORT is not serving this step's page — another process holds it (set HTTP_PORT=<free port>)"
    ;;
esac

if [ -n "$URL_MD" ]; then
  ok "the page was fetched and converted"
  check "navigation and footer are gone" \
    test "$(grep -cE 'Home About Contact|Example Corp' "$URL_MD")" -eq 0
  check "author and title survive as metadata" grep -q '^author: Ops Team' "$URL_MD"
  check "the table survives as a table" grep -q '| Region' "$URL_MD"
  show "what came back" "$URL_MD"
  file "the page that was served" 00-inputs/site/index.html
elif [ -n "$SERVED" ]; then
  bad "the fetch produced no markdown"
  file "fetch log" 05-url/fetch.log
fi
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""

# ── 6 ────────────────────────────────────────────────────────────────────────
step 06-batch "A directory of $BATCH_DOCS documents" \
  "Throughput, concurrency, and the number a manager asks about first: what it cost."
mkdir -p 00-inputs/batch-docs
i=1
while [ "$i" -le "$BATCH_DOCS" ]; do
  {
    echo "# Doc $i"; echo
    j=1
    while [ "$j" -le 40 ]; do
      echo "Paragraph $j of document $i, with   messy   spacing to clean up."; echo
      j=$((j + 1))
    done
  } >"00-inputs/batch-docs/doc$i.md"
  i=$((i + 1))
done
MODEL="$E2E_MODEL" markitai 00-inputs/batch-docs/ -o 06-batch/output/ --llm --no-cache \
  >06-batch/batch.log 2>&1
DONE=$(ls 06-batch/output/*.llm.md 2>/dev/null | wc -l | tr -d ' ')
check "every document was enhanced ($DONE of $BATCH_DOCS)" test "$DONE" -eq "$BATCH_DOCS"
check "the run reports what it spent" grep -qE '\$[0-9]' 06-batch/batch.log
SUMMARY_LINE=$(grep -oE 'Done:.*' 06-batch/batch.log | head -1)
[ -n "$SUMMARY_LINE" ] && note "$SUMMARY_LINE"
file "run log" 06-batch/batch.log

# ── 7 ────────────────────────────────────────────────────────────────────────
step 07-interrupt-resume "Stopping half way, and picking up again" \
  "The question behind it: does an interrupted run cost you the work already paid for?"
if [ "$SKIP_INTERRUPT" = "1" ]; then
  skip "skipped (SKIP_INTERRUPT=1)"
else
  # The script sends the interrupt rather than asking for Ctrl-C. Ctrl-C goes
  # to the whole foreground process group, so it would take this script down
  # with markitai and the resume half would never run. What is checked is
  # markitai's behaviour on SIGINT, which is the same either way.
  #
  # The launcher exists because a shell that is not interactive starts its
  # background children with SIGINT already ignored, and Python keeps an
  # inherited SIG_IGN — signal markitai without this and it runs to
  # completion, which reads convincingly like "Ctrl-C does nothing".
  cat >_internal/interrupt_launcher.py <<'PYEOF'
import os
import signal
import sys

signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
PYEOF

  # Completions reach the state file on an interval (10s by default), so a
  # batch finishing in under ~20s can only be interrupted inside that window,
  # with nothing recorded to resume from. Shorten the interval for these two
  # runs instead of generating enough documents to outlast it: same
  # machinery, a fraction of the spend. Both runs pass the same override so
  # they agree on the state file.
  FLUSH_OVERRIDE='{"batch":{"state_flush_interval_seconds":2}}'

  MODEL="$E2E_MODEL" python3 _internal/interrupt_launcher.py \
    markitai 00-inputs/batch-docs/ -o 07-interrupt-resume/output/ --llm --no-cache \
    --config-json "$FLUSH_OVERRIDE" >07-interrupt-resume/interrupted.log 2>&1 &
  RUN_PID=$!
  sleep "$INTERRUPT_AFTER"
  kill -INT "$RUN_PID" 2>/dev/null
  wait "$RUN_PID" 2>/dev/null
  RUN_RC=$?
  RUN_PID=""

  PARTIAL=$(ls 07-interrupt-resume/output/*.llm.md 2>/dev/null | wc -l | tr -d ' ')
  if [ "$PARTIAL" -ge "$BATCH_DOCS" ]; then
    skip "the batch finished inside ${INTERRUPT_AFTER}s — raise BATCH_DOCS or lower INTERRUPT_AFTER"
  elif [ "$PARTIAL" -eq 0 ]; then
    skip "nothing had finished at ${INTERRUPT_AFTER}s — raise INTERRUPT_AFTER"
  else
    note "interrupted after ${INTERRUPT_AFTER}s with $PARTIAL of $BATCH_DOCS written"
    check "the interrupt stops the run" test "$RUN_RC" -ne 0
    check "it says how to continue instead of leaving you guessing" \
      grep -q -- '--resume' 07-interrupt-resume/interrupted.log
    check "the progress it had made survived on disk" \
      test -n "$(ls 07-interrupt-resume/output/.markitai/states/*.state.json 2>/dev/null)"

    MODEL="$E2E_MODEL" markitai 00-inputs/batch-docs/ -o 07-interrupt-resume/output/ \
      --llm --no-cache --resume --config-json "$FLUSH_OVERRIDE" \
      >07-interrupt-resume/resumed.log 2>&1
    check "resume reads the interrupted run's state" \
      grep -q 'Resuming batch' 07-interrupt-resume/resumed.log
    RESUME_LINE=$(grep -oE 'Resuming batch.*' 07-interrupt-resume/resumed.log | head -1)
    [ -n "$RESUME_LINE" ] && note "$RESUME_LINE"

    # The line alone is not the point — a resume reporting 0 completed has
    # restarted, and the user pays for the whole batch again.
    CARRIED=$(grep -oE 'Resuming batch: [0-9]+' 07-interrupt-resume/resumed.log \
      | grep -oE '[0-9]+' | head -1)
    CARRIED=${CARRIED:-0}
    if [ "$CARRIED" -gt 0 ]; then
      ok "work already paid for is skipped, not redone ($CARRIED documents)"
    else
      skip "resume carried nothing over: nothing had been flushed when the interrupt landed — raise INTERRUPT_AFTER"
    fi
    check "and the batch finishes" \
      test "$(ls 07-interrupt-resume/output/*.llm.md 2>/dev/null | wc -l | tr -d ' ')" -eq "$BATCH_DOCS"
    file "the interrupted run" 07-interrupt-resume/interrupted.log
    file "the resumed run" 07-interrupt-resume/resumed.log
  fi
fi

# ── 8 ────────────────────────────────────────────────────────────────────────
step 08-cache "Paying once" \
  "Re-running the same work should cost nothing: the difference between a tool you can iterate with and one you cannot."
MODEL="$E2E_MODEL" markitai 00-inputs/batch-docs/ -o 08-cache/first-run/ --llm \
  >08-cache/first-run.log 2>&1
START=$(date +%s)
MODEL="$E2E_MODEL" markitai 00-inputs/batch-docs/ -o 08-cache/second-run/ --llm \
  >08-cache/second-run.log 2>&1
ELAPSED=$(( $(date +%s) - START ))
check "the repeat run is served from cache" grep -qE 'Cache: [0-9]+' 08-cache/second-run.log
check "and returns in ${ELAPSED}s" test "$ELAPSED" -le 10
file "second run log" 08-cache/second-run.log

# ── 9 ────────────────────────────────────────────────────────────────────────
step 09-failure-messages "The messages you meet on a bad day" \
  "A tool is judged on its errors more than its successes: they arrive when someone is already stuck."
markitai /definitely/not/here.txt >09-failure-messages/missing-file.txt 2>&1
check "a missing file exits non-zero rather than pretending" test $? -ne 0
note "$(head -1 09-failure-messages/missing-file.txt)"

# An image, not the PDF fixture: a born-digital PDF has a text layer, so --ocr
# on it is a no-op that succeeds and never reaches the missing backend.
python3 - <<'PNG'
import base64, pathlib
pathlib.Path("00-inputs/probe.png").write_bytes(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAFklEQVR4nGP8//8/AzGAiShVowZS"
    "z0AAQFgBBSBqfMcAAAAASUVORK5CYII="))
PNG
markitai 00-inputs/probe.png --ocr -o 09-failure-messages/ocr/ \
  >09-failure-messages/missing-extra.txt 2>&1
if grep -qE '(uv tool|pipx|pip) install' 09-failure-messages/missing-extra.txt; then
  ok "asking for a capability that is not installed names the command that installs it"
  note "$(grep -oE '(uv tool|pipx|pip) install [^ ]*( --force)?' 09-failure-messages/missing-extra.txt | head -1)"
elif grep -qi 'rapidocr' 09-failure-messages/missing-extra.txt; then
  skip "the ocr extra is present in this environment; nothing to report"
else
  bad "--ocr without its backend did not explain itself"
fi
show "the message when a capability is missing" 09-failure-messages/missing-extra.txt

# ── 10 ───────────────────────────────────────────────────────────────────────
step 10-formats "Every format the front page lists" \
  "One directory of one file per format, all converted by the base wheel."
mkdir -p 00-inputs/formats
# Every documented format is handled by the base wheel; images are skipped
# without --llm/--ocr, and the summary has to say so.
BASE_FORMATS="pptx xlsx csv tsv html xml eml msg epub ipynb rst org tex odt ods rtf"
IMAGE_FORMATS="jpg bmp"
for f in $BASE_FORMATS $IMAGE_FORMATS; do
  cp "$REPO_ROOT/packages/markitai/tests/fixtures/sample.$f" 00-inputs/formats/ 2>/dev/null
done
cp "$REPO_ROOT/packages/markitai/tests/fixtures/legacy/sample.xls" 00-inputs/formats/ 2>/dev/null
markitai 00-inputs/formats/ -o 10-formats/output/ >10-formats/batch.log 2>&1
MISSING=""
for f in $BASE_FORMATS xls; do
  [ -s "10-formats/output/sample.$f.md" ] || MISSING="$MISSING $f"
done
check "every base-wheel format produced markdown${MISSING:+ (missing:$MISSING)}" test -z "$MISSING"
check "images are skipped, and the summary says what would read them" \
  grep -q 'image_only.*--llm or --ocr' 10-formats/batch.log
check "no format is left without a converter" \
  test "$(grep -c 'No converter available' 10-formats/batch.log)" -eq 0
check "an RTF keeps its heading hierarchy" \
  bash -c "grep -q '^# ' 10-formats/output/sample.rtf.md && grep -q '^## ' 10-formats/output/sample.rtf.md && grep -q '^### ' 10-formats/output/sample.rtf.md"
check "a spreadsheet keeps its table" grep -q '^|' 10-formats/output/sample.xlsx.md
check "a TSV becomes a table" grep -q '^| Employee_ID' 10-formats/output/sample.tsv.md
check "an ODS sheet becomes a table" grep -q '^| Team' 10-formats/output/sample.ods.md
check "an ODT keeps its heading and list" \
  bash -c "grep -q '^# ' 10-formats/output/sample.odt.md && grep -q '^- ' 10-formats/output/sample.odt.md"
check "an XML document keeps its element names and text" \
  bash -c "grep -q 'Tove' 10-formats/output/sample.xml.md && grep -qE '^#+ (note|to)' 10-formats/output/sample.xml.md"
check "an RST title becomes a heading" grep -q '^# Example Docs' 10-formats/output/sample.rst.md
check "an Org title becomes a heading" grep -q '^# Example Docs' 10-formats/output/sample.org.md
check "a TeX body loses its preamble and keeps its text" \
  bash -c "grep -q 'Hello World from LaTeX' 10-formats/output/sample.tex.md && ! grep -q documentclass 10-formats/output/sample.tex.md"
check "an email keeps its headers" grep -q '^\*\*From:\*\*' 10-formats/output/sample.eml.md
check "a notebook keeps its code cells" grep -q '```' 10-formats/output/sample.ipynb.md
check "an HTML file converts to stdout like any other" \
  test "$(markitai 00-inputs/formats/sample.html 2>/dev/null | head -c 3)" = "---"
show "the batch summary" 10-formats/batch.log

# ── 11 ───────────────────────────────────────────────────────────────────────
step 11-presets-profiles "Presets and output profiles" \
  "A preset decides how much the model does; a profile decides who the output is shaped for."
markitai 00-inputs/sample.pdf -o 11-presets-profiles/minimal/ --preset minimal \
  >11-presets-profiles/minimal.log 2>&1
check "--preset minimal converts without touching a model" \
  test -f 11-presets-profiles/minimal/sample.pdf.md -a ! -f 11-presets-profiles/minimal/sample.pdf.llm.md
MODEL="$E2E_MODEL" markitai 00-inputs/sample.pdf -o 11-presets-profiles/standard-no-desc/ \
  --preset standard --no-desc >11-presets-profiles/standard.log 2>&1
check "--preset standard enhances with alt text" \
  grep -qE '!\[[^]]+\]\(' 11-presets-profiles/standard-no-desc/sample.pdf.llm.md
check "and --no-desc on top of it leaves the descriptions sidecar out" \
  test ! -f 11-presets-profiles/standard-no-desc/.markitai/assets/images.json
markitai 00-inputs/sample.pdf -o 11-presets-profiles/rag/ --profile rag >11-presets-profiles/rag.log 2>&1
check "--profile rag puts images in a visible assets/ directory" \
  test -d 11-presets-profiles/rag/assets
check "and references them there" grep -q '](assets/' 11-presets-profiles/rag/sample.pdf.md
markitai 00-inputs/sample.pdf -o 11-presets-profiles/obsidian/ --profile obsidian \
  --config-json '{"output":{"wikilinks":true}}' >11-presets-profiles/obsidian.log 2>&1
check "--profile obsidian with wikilinks writes ![[assets/...]]" \
  grep -q '!\[\[assets/' 11-presets-profiles/obsidian/sample.pdf.md
markitai 00-inputs/sample.docx -o 11-presets-profiles/okf/ --profile okf >11-presets-profiles/okf.log 2>&1
check "--profile okf writes Open Knowledge Format frontmatter" \
  grep -q '^type: Document' 11-presets-profiles/okf/sample.docx.md
check "including who generated it and when" \
  grep -qE "^  by: markitai/$VERSION" 11-presets-profiles/okf/sample.docx.md
show "OKF frontmatter" 11-presets-profiles/okf/sample.docx.md
file "rag output" 11-presets-profiles/rag/sample.pdf.md

# ── 12 ───────────────────────────────────────────────────────────────────────
step 12-run-shape "Flags that change the shape of a run" \
  "Preview, pure, keep-base, screenshots, discovery filters, and reading an image with the model."
markitai 00-inputs/sample.docx -o 12-run-shape/dry-run/ --dry-run >12-run-shape/dry-run.txt 2>&1
check "--dry-run names the output it would write" grep -q 'sample.docx.md' 12-run-shape/dry-run.txt
check "and writes nothing" test ! -e 12-run-shape/dry-run/sample.docx.md
check "--pure yields the body alone, no frontmatter" \
  test "$(markitai 00-inputs/sample.docx --pure 2>/dev/null | head -c 1)" = "#"
MODEL="$E2E_MODEL" markitai 00-inputs/sample.docx -o 12-run-shape/keep-base/ --llm --keep-base \
  >12-run-shape/keep-base.log 2>&1
check "--keep-base leaves the base .md beside the .llm.md" \
  test -f 12-run-shape/keep-base/sample.docx.md -a -f 12-run-shape/keep-base/sample.docx.llm.md
markitai 00-inputs/sample.pdf -o 12-run-shape/screenshot/ --screenshot >12-run-shape/screenshot.log 2>&1
check "--screenshot renders one image per PDF page" \
  test "$(ls 12-run-shape/screenshot/.markitai/screenshots/*.jpg 2>/dev/null | wc -l | tr -d ' ')" -ge 1
mkdir -p 00-inputs/tree/a/b
cp 00-inputs/formats/sample.csv 00-inputs/tree/
cp 00-inputs/formats/sample.tsv 00-inputs/tree/a/
cp 00-inputs/formats/sample.xml 00-inputs/tree/a/b/
markitai 00-inputs/tree/ -o 12-run-shape/glob/ --glob '*.csv' >12-run-shape/glob.log 2>&1
check "--glob restricts discovery to matching files" \
  test "$(ls 12-run-shape/glob/*.md 2>/dev/null | wc -l | tr -d ' ')" -eq 1
markitai 00-inputs/tree/ -o 12-run-shape/depth/ --max-depth 0 >12-run-shape/depth.log 2>&1
check "--max-depth 0 stays in the input directory" \
  test "$(ls 12-run-shape/depth/*.md 2>/dev/null | wc -l | tr -d ' ')" -eq 1
MODEL="$E2E_MODEL" markitai 00-inputs/formats/sample.jpg -o 12-run-shape/vlm-ocr/ --llm --ocr \
  >12-run-shape/vlm-ocr.log 2>&1
check "--llm --ocr on an image has the vision model read it" \
  grep -q '^description:' 12-run-shape/vlm-ocr/sample.jpg.llm.md
show "dry-run preview" 12-run-shape/dry-run.txt
file "what the model saw in the image" 12-run-shape/vlm-ocr/sample.jpg.llm.md

# ── 13 ───────────────────────────────────────────────────────────────────────
step 13-refusals "Refusals that should be refusals" \
  "Removed flags, private URLs offered to remote services, and Batch API on the wrong pool."
markitai 00-inputs/sample.docx --playwright >13-refusals/removed-alias.txt 2>&1
ALIAS_RC=$?
check "a removed alias is a usage error, not a silent no-op" test "$ALIAS_RC" -ne 0
check "and the error names the replacement" grep -q -- '-s playwright' 13-refusals/removed-alias.txt
markitai "http://127.0.0.1:$HTTP_PORT/" -s jina -o 13-refusals/jina/ >13-refusals/jina-local.txt 2>&1
check "a local URL is never offered to a remote extraction service" \
  grep -qi 'private/local host' 13-refusals/jina-local.txt
markitai 00-inputs/sample.docx -o 13-refusals/llm-batch-file/ --llm --llm-batch \
  >13-refusals/llm-batch-file.txt 2>&1
check "--llm-batch on a single file is refused as directory-only" \
  grep -q 'directory batches only' 13-refusals/llm-batch-file.txt
MODEL="$E2E_MODEL" markitai 00-inputs/batch-docs/ -o 13-refusals/llm-batch-pool/ --llm --llm-batch \
  >13-refusals/llm-batch-pool.txt 2>&1
check "--llm-batch on a pool without a Batch API says which pools have one" \
  grep -q 'openai and anthropic' 13-refusals/llm-batch-pool.txt
markitai 00-inputs/sample.docx --preset nope >13-refusals/bad-preset.txt 2>&1
check "an unknown preset is rejected" test $? -ne 0
show "what a removed flag tells you" 13-refusals/removed-alias.txt
show "what a remote strategy says about a private URL" 13-refusals/jina-local.txt

# ── 14 ───────────────────────────────────────────────────────────────────────
step 14-url-list "A .urls list" \
  "The batch form of the URL step: one file, one line per page."
cat >00-inputs/site/second.html <<'HTML'
<!doctype html><html><head><title>Second Page</title></head><body>
<article><h1>Second Page</h1>
<p>A second article, long enough to read as content rather than an empty shell:
it repeats the point of the first one in different words and then adds a
closing paragraph so the extractor has something to keep.</p>
<p>Closing paragraph with a <a href="/">link back</a> and a final sentence.</p>
</article></body></html>
HTML
(cd 00-inputs/site && exec python3 -m http.server "$HTTP_PORT" >/dev/null 2>&1) &
SERVER_PID=$!
for _ in 1 2 3 4 5; do
  sleep 1
  curl -fsS --max-time 2 "http://127.0.0.1:$HTTP_PORT/second.html" 2>/dev/null | grep -q 'Second Page' && break
done
printf 'http://127.0.0.1:%s/\n\nhttp://127.0.0.1:%s/second.html\n' "$HTTP_PORT" "$HTTP_PORT" >00-inputs/pages.urls
markitai 00-inputs/pages.urls -o 14-url-list/output/ >14-url-list/run.log 2>&1
URLS_OUT=$(ls 14-url-list/output/*.md 2>/dev/null | wc -l | tr -d ' ')
check "both pages in the list were converted ($URLS_OUT of 2)" test "$URLS_OUT" -eq 2
check "blank lines in the list are ignored, not fetched" \
  test "$(grep -c '✗' 14-url-list/run.log)" -eq 0
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
file "run log" 14-url-list/run.log

# ── 15 ───────────────────────────────────────────────────────────────────────
step 15-config "Configuration, three layers" \
  "A user file, a project file, and inline JSON — and the commands that read and write them."
markitai init --yes >15-config/init.txt 2>&1
check "init --yes writes the user config without asking" test -f "$HOME/.markitai/config.json"
markitai config set llm.enabled true >15-config/set.txt 2>&1
check "config set persists a value" test "$(markitai config get llm.enabled 2>/dev/null)" = "True"
markitai config validate >15-config/validate.txt 2>&1
check "config validate accepts what config set wrote" test $? -eq 0
markitai config path >15-config/path.txt 2>&1
check "config path explains precedence" grep -qi 'priority' 15-config/path.txt
(cd 15-config && markitai init --local --yes >local-init.txt 2>&1)
check "init --local writes a project markitai.json" test -f 15-config/markitai.json
markitai 00-inputs/sample.docx -o 15-config/override/ --dry-run \
  --config-json '{"llm":{"enabled":false}}' >15-config/override.txt 2>&1
check "--config-json overrides the file the user just wrote" \
  test "$(grep -c 'Features: none' 15-config/override.txt)" -eq 1
# init --yes wrote a model pool from the detected keys, and MODEL only pins a
# model when no pool is configured — so the file goes away again here, and
# later steps run on the bare home they expect.
rm -f "$HOME/.markitai/config.json"
markitai cache stats >15-config/cache-stats.txt 2>&1
check "cache stats reports what the earlier runs left behind" grep -qE 'entries' 15-config/cache-stats.txt
markitai cache clear -y >15-config/cache-clear.txt 2>&1
check "cache clear -y needs no confirmation" test $? -eq 0
show "configuration precedence, as the tool explains it" 15-config/path.txt
file "cache statistics" 15-config/cache-stats.txt

# ── 16 ───────────────────────────────────────────────────────────────────────
step 16-python-api "markitai as a library" \
  "The same conversion from Python: a typed result, a clean stdout, and diagnostics that can be silenced."
TOOL_PY="$UV_TOOL_DIR/markitai/bin/python"
cat >_internal/api_probe.py <<'PYEOF'
import sys
from loguru import logger
import markitai

if "--quiet" in sys.argv:
    logger.disable("markitai")
out = markitai.convert(sys.argv[1])
print(type(out).__name__, out.frontmatter.get("title"), len(out.markdown))
PYEOF
"$TOOL_PY" _internal/api_probe.py 00-inputs/sample.docx >16-python-api/stdout.txt 2>16-python-api/stderr.txt
check "convert() returns a ConversionOutput with the parsed frontmatter" \
  grep -q '^ConversionOutput Markitai Snapshot Fixture [0-9]' 16-python-api/stdout.txt
check "stdout carries only what the program printed" \
  test "$(wc -l <16-python-api/stdout.txt | tr -d ' ')" -eq 1
"$TOOL_PY" _internal/api_probe.py 00-inputs/sample.docx --quiet >/dev/null 2>16-python-api/stderr-quiet.txt
check "logger.disable(\"markitai\") silences the diagnostics, as documented" \
  test ! -s 16-python-api/stderr-quiet.txt
file "diagnostics the library emits by default" 16-python-api/stderr.txt

# ── 17 ───────────────────────────────────────────────────────────────────────
step 17-extras "The optional extras, installed" \
  "serve, mcp and legacy added to the same install; doctor notices, and the formats they unlock convert."
uv tool install --force --python "$E2E_PYTHON" "markitai[serve,mcp,legacy]@$WHEEL" \
  >17-extras/install.log 2>&1
check "the extras install on top of the existing tool" test $? -eq 0
markitai doctor --json >17-extras/doctor.json 2>/dev/null
python3 - <<'PYEOF' >17-extras/doctor-extras.txt
import json
d = json.load(open("17-extras/doctor.json"))
for key in ("serve", "anydoc"):
    print(key, d.get(key, {}).get("status"))
PYEOF
check "doctor sees the serve extra" grep -qE '^serve (ok|installed|available)' 17-extras/doctor-extras.txt
check "doctor sees the legacy backend" grep -qE '^anydoc (ok|installed|available)' 17-extras/doctor-extras.txt
cp "$REPO_ROOT/packages/markitai/tests/fixtures/legacy/sample.doc" \
   "$REPO_ROOT/packages/markitai/tests/fixtures/legacy/sample.ppt" 00-inputs/
markitai 00-inputs/sample.doc -o 17-extras/legacy/ >17-extras/doc.log 2>&1
markitai 00-inputs/sample.ppt -o 17-extras/legacy/ >17-extras/ppt.log 2>&1
check "a Word 97 .doc converts through the legacy extra" test -s 17-extras/legacy/sample.doc.md
check "a PowerPoint 97 .ppt converts through the legacy extra" test -s 17-extras/legacy/sample.ppt.md
show "doctor after the extras" 17-extras/doctor-extras.txt
file "legacy .doc output" 17-extras/legacy/sample.doc.md

# ── 18 ───────────────────────────────────────────────────────────────────────
step 18-serve "The web workspace" \
  "The same API the browser UI talks to: capabilities, the packaged UI, one job to completion, history."
cat >_internal/serve_probe.py <<'PYEOF'
import json, sys, time, urllib.request, urllib.error, uuid

base, doc = sys.argv[1].rstrip("/"), sys.argv[2]

def call(method, path, body=None, headers=None, raw=False):
    req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data or b"null"))
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode(errors="replace")

def multipart(fields, files):
    b = uuid.uuid4().hex; out = bytearray()
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data) in files.items():
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                f'Content-Type: application/octet-stream\r\n\r\n').encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), {"Content-Type": f"multipart/form-data; boundary={b}"}

r = {}
st, caps = call("GET", "/api/capabilities")
r["capabilities_status"] = st
r["version"] = caps.get("version") if isinstance(caps, dict) else None
r["presets"] = ",".join(caps.get("presets", [])) if isinstance(caps, dict) else None
st, html = call("GET", "/", headers={"Accept": "text/html"}, raw=True)
r["ui_is_html"] = st == 200 and b"<!doctype html" in html.lower()
body, hdr = multipart({"urls": "[]", "options": json.dumps({"llm": False})},
                      {"files": (doc.rsplit("/", 1)[-1], open(doc, "rb").read())})
st, created = call("POST", "/api/jobs", body, hdr)
r["create_status"] = st
job_id = created["job_id"] if isinstance(created, dict) else None
snapshot = {}
for _ in range(120):
    st, snapshot = call("GET", f"/api/jobs/{job_id}")
    if isinstance(snapshot, dict) and snapshot.get("status") == "done":
        break
    time.sleep(0.5)
items = snapshot.get("items", []) if isinstance(snapshot, dict) else []
r["job_status"] = snapshot.get("status") if isinstance(snapshot, dict) else None
r["item_status"] = items[0]["status"] if items else None
if items:
    st, result = call("GET", f"/api/jobs/{job_id}/items/{items[0]['item_id']}/result")
    r["result_starts_with_frontmatter"] = isinstance(result, dict) and (result.get("markdown") or "").startswith("---")
st, history = call("GET", "/api/history")
r["history_entries"] = len(history) if isinstance(history, list) else -1
r["history_has_job"] = isinstance(history, list) and any(h.get("job_id") == job_id for h in history)
for k, v in r.items():
    print(f"{k}={v}")
PYEOF
# A CLI run recorded into the shared history should be visible from the UI.
markitai 00-inputs/sample.docx -o 18-serve/recorded/ --record-history >18-serve/recorded.log 2>&1
markitai serve --no-open --port "$SERVE_PORT" >18-serve/serve.log 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do
  curl -fsS --max-time 2 "http://127.0.0.1:$SERVE_PORT/api/capabilities" >/dev/null 2>&1 && break
  sleep 1
done
python3 _internal/serve_probe.py "http://127.0.0.1:$SERVE_PORT" 00-inputs/sample.docx \
  >18-serve/probe.txt 2>18-serve/probe.err
check "the server comes up and reports the installed version" \
  grep -q "^version=$VERSION$" 18-serve/probe.txt
check "the sign-in URL is printed for the user" grep -q '?token=' 18-serve/serve.log
check "the packaged UI is served at /" grep -q '^ui_is_html=True' 18-serve/probe.txt
check "the capabilities list the three presets" grep -q '^presets=minimal,standard,rich' 18-serve/probe.txt
check "an uploaded file becomes a finished job" grep -q '^item_status=done' 18-serve/probe.txt
check "and its result is the same markdown the CLI writes" \
  grep -q '^result_starts_with_frontmatter=True' 18-serve/probe.txt
check "history lists the job" grep -q '^history_has_job=True' 18-serve/probe.txt
check "and the CLI run recorded with --record-history" \
  test "$(grep -oE '^history_entries=[0-9]+' 18-serve/probe.txt | grep -oE '[0-9]+')" -ge 2
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
show "what the API reported" 18-serve/probe.txt
file "server log" 18-serve/serve.log

# ── 19 ───────────────────────────────────────────────────────────────────────
step 19-mcp "markitai-mcp over stdio" \
  "What an agent host sees: a handshake, four tools, and one conversion through the protocol."
cat >_internal/mcp_probe.py <<'PYEOF'
import json, subprocess, sys

cmd, doc, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
proc = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
seq = 0

def send(method, params=None, notify=False):
    global seq
    msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not notify:
        seq += 1
        msg["id"] = seq
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if notify:
        return None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(f"server closed stdout; stderr: {proc.stderr.read()[-800:]}")
        reply = json.loads(line)
        if reply.get("id") == seq:
            return reply

init = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "e2e", "version": "0"}})
send("notifications/initialized", notify=True)
tools = send("tools/list")
call = send("tools/call", {"name": "convert_document",
                           "arguments": {"path": doc, "output_dir": out_dir}})
proc.stdin.close()
proc.wait(timeout=30)
info = init.get("result", {}).get("serverInfo", {})
content = call.get("result", {}).get("structuredContent") or {}
print("server=" + info.get("name", "") + " " + info.get("version", ""))
print("tools=" + ",".join(sorted(t["name"] for t in tools.get("result", {}).get("tools", []))))
print("is_error=" + str(call.get("result", {}).get("isError", False)))
print("markdown_file=" + str(content.get("markdown_file")))
PYEOF
mkdir -p 19-mcp/output
python3 _internal/mcp_probe.py "$(command -v markitai-mcp)" "$PWD/00-inputs/sample.docx" "$PWD/19-mcp/output" \
  >19-mcp/probe.txt 2>19-mcp/probe.err
check "the server completes the handshake and identifies itself" \
  grep -q "^server=markitai $VERSION$" 19-mcp/probe.txt
check "it exposes exactly the four documented tools" \
  grep -q '^tools=batch_convert,convert_document,convert_url,job_status$' 19-mcp/probe.txt
check "convert_document converts through the protocol" grep -q '^is_error=False' 19-mcp/probe.txt
check "and the file it names exists" test -s "$(grep -oE '^markdown_file=.*' 19-mcp/probe.txt | cut -d= -f2-)"
check "markitai mcp is the same server as a CLI subcommand (the registry's form)" \
  markitai mcp --help
show "what the agent host saw" 19-mcp/probe.txt

# ── 20 ───────────────────────────────────────────────────────────────────────
step 20-all-extras "Every installable extra, and the browser" \
  "browser and extra-fetch join the install; doctor --fix fetches Chromium the way the docs say."
uv tool install --force --python "$E2E_PYTHON" \
  "markitai[serve,mcp,legacy,browser,extra-fetch]@$WHEEL" >20-all-extras/install.log 2>&1
check "the remaining extras install on top of the existing tool" test $? -eq 0
"$TOOL_PY" -c "import curl_cffi, playwright" >20-all-extras/imports.txt 2>&1
check "curl_cffi and playwright import from the tool environment" test $? -eq 0
note "browsers live in $PLAYWRIGHT_BROWSERS_PATH (set it to an empty dir to exercise the cold download)"
markitai doctor --fix >20-all-extras/doctor-fix.txt 2>&1
markitai doctor --json >20-all-extras/doctor.json 2>/dev/null
python3 - <<'PYEOF' >20-all-extras/doctor-playwright.txt
import json
d = json.load(open("20-all-extras/doctor.json"))
print("playwright", d.get("playwright", {}).get("status"), "-", d.get("playwright", {}).get("message", ""))
PYEOF
check "doctor --fix leaves a working Chromium behind" grep -q '^playwright ok' 20-all-extras/doctor-playwright.txt
show "doctor --fix, as the user sees it" 20-all-extras/doctor-fix.txt
file "tool install log" 20-all-extras/install.log

# ── 22 ───────────────────────────────────────────────────────────────────────
step 22-browser "Fetching with a real browser" \
  "Playwright rendering, a full-page screenshot, and reading a page from its screenshot alone."
(cd 00-inputs/site && exec python3 -m http.server "$HTTP_PORT" >/dev/null 2>&1) &
SERVER_PID=$!
for _ in 1 2 3 4 5; do
  sleep 1
  curl -fsS --max-time 2 "http://127.0.0.1:$HTTP_PORT/" 2>/dev/null | grep -q 'Quarterly Report' && break
done
markitai "http://127.0.0.1:$HTTP_PORT/" -s playwright -o 22-browser/playwright/ >22-browser/playwright.log 2>&1
PW_MD=$(ls 22-browser/playwright/*.md 2>/dev/null | head -1)
check "-s playwright renders the page in Chromium" test -n "$PW_MD"
check "and says so in the frontmatter" grep -q '^fetch_strategy: playwright' "${PW_MD:-/dev/null}"
check "the article survives the browser route too" grep -q '| Region' "${PW_MD:-/dev/null}"
markitai "http://127.0.0.1:$HTTP_PORT/" --screenshot -o 22-browser/screenshot/ >22-browser/screenshot.log 2>&1
check "--screenshot on a URL saves a full-page capture" \
  test -n "$(ls 22-browser/screenshot/.markitai/screenshots/*.full.jpg 2>/dev/null)"
MODEL="$E2E_MODEL" markitai "http://127.0.0.1:$HTTP_PORT/" --screenshot-only --llm \
  -o 22-browser/screenshot-only/ >22-browser/screenshot-only.log 2>&1
SO_MD=$(ls 22-browser/screenshot-only/*.llm.md 2>/dev/null | head -1)
check "--screenshot-only --llm has the vision model read the page from its screenshot" \
  grep -q '^description:' "${SO_MD:-/dev/null}"
check "and the output points back at the screenshot it was read from" \
  grep -q 'Screenshot for reference' "${SO_MD:-/dev/null}"
check "what it read matches the page" grep -qi 'Region B' "${SO_MD:-/dev/null}"
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""
show "the page as the vision model read it" "${SO_MD:-22-browser/screenshot-only.log}"
file "playwright route output" "${PW_MD:-22-browser/playwright.log}"

# ── 23 ───────────────────────────────────────────────────────────────────────
step 23-cloudflare-backend "File conversion through Cloudflare" \
  "-b cloudflare sends the file to Workers AI toMarkdown; the API host is reached even where URL targets cannot be."
if [ -n "${CLOUDFLARE_API_TOKEN:-}" ] && [ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  markitai 00-inputs/sample.docx -b cloudflare -o 23-cloudflare-backend/output/ -v \
    >23-cloudflare-backend/convert.log 2>&1
  check "the document converts through Cloudflare" test -s 23-cloudflare-backend/output/sample.docx.md
  check "and the run says which converter it used" grep -qi 'Cloudflare toMarkdown' 23-cloudflare-backend/convert.log
  file "conversion log" 23-cloudflare-backend/convert.log
else
  skip "CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID not in $ENV_FILE"
fi

# ── 24 ───────────────────────────────────────────────────────────────────────
step 24-remote-strategies "Remote extraction services" \
  "jina, defuddle and Cloudflare Browser Rendering on a public page — each needs its key, and a resolver that tells public from private."
RESOLVED=$(python3 - "$E2E_PUBLIC_URL" <<'PYEOF'
import ipaddress, socket, sys
from urllib.parse import urlsplit
host = urlsplit(sys.argv[1]).hostname or ""
try:
    ip = ipaddress.ip_address(socket.gethostbyname(host))
except OSError as e:
    print(f"unresolvable ({e})"); raise SystemExit
fake = ipaddress.ip_network("198.18.0.0/15")
print("public" if ip.is_global and ip not in fake else f"non-public ({ip})")
PYEOF
)
note "$E2E_PUBLIC_URL resolves as: $RESOLVED"
if [ "$RESOLVED" != "public" ]; then
  skip "this machine's resolver maps public hosts to a non-public address (a fake-IP VPN, typically); markitai correctly refuses to hand such URLs to remote services — run with the VPN's TUN mode off to exercise them"
else
  for STRATEGY in jina defuddle cloudflare; do
    case "$STRATEGY" in
      jina) [ -n "${JINA_API_KEY:-}" ] || { skip "-s jina: JINA_API_KEY not in $ENV_FILE"; continue; } ;;
      cloudflare) [ -n "${CLOUDFLARE_API_TOKEN:-}" ] || { skip "-s cloudflare: CLOUDFLARE_API_TOKEN not in $ENV_FILE"; continue; } ;;
    esac
    markitai "$E2E_PUBLIC_URL" -s "$STRATEGY" -o "24-remote-strategies/$STRATEGY/" \
      >"24-remote-strategies/$STRATEGY.log" 2>&1
    R_MD=$(ls "24-remote-strategies/$STRATEGY"/*.md 2>/dev/null | head -1)
    check "-s $STRATEGY fetches the page through the service" test -n "$R_MD"
    check "and the frontmatter records that route" grep -q "^fetch_strategy: $STRATEGY" "${R_MD:-/dev/null}"
    check "with a real article body behind it" test "$(wc -c <"${R_MD:-/dev/null}" | tr -d ' ')" -gt 500
    file "-s $STRATEGY output" "${R_MD:-24-remote-strategies/$STRATEGY.log}"
  done
fi

# ── 25 ───────────────────────────────────────────────────────────────────────
step 25-batch-api "The Batch API, at half price" \
  "A real --llm-batch job: submitted, waited for, and collected later if it outlasts the wait."
BATCH_PROVIDER=${E2E_BATCH_MODEL%%/*}
case "$BATCH_PROVIDER" in
  openai) BATCH_KEY=${OPENAI_API_KEY:-} ;;
  anthropic) BATCH_KEY=${ANTHROPIC_API_KEY:-} ;;
  *) BATCH_KEY="" ;;
esac
if [ -z "$BATCH_KEY" ]; then
  skip "E2E_BATCH_MODEL=$E2E_BATCH_MODEL has no key in $ENV_FILE"
else
  mkdir -p 00-inputs/batch-api
  for i in 1 2 3; do
    printf '# Note %s\n\nSome   messy   text  to clean, paragraph one.\n\nParagraph two of note %s.\n' "$i" "$i" \
      >"00-inputs/batch-api/note$i.md"
  done
  MODEL="$E2E_BATCH_MODEL" markitai 00-inputs/batch-api/ -o 25-batch-api/output/ --llm --no-cache \
    --llm-batch --llm-batch-timeout 120 >25-batch-api/submit.log 2>&1
  BATCH_RC=$?
  if [ "$BATCH_RC" -eq 0 ]; then
    ok "the batch completed inside the wait"
  elif [ "$BATCH_RC" -eq 2 ]; then
    ok "past the wait, the run hands off instead of blocking"
    check "and names the exact collect command" grep -q -- '--llm-batch-collect' 25-batch-api/submit.log
    BATCH_ID=$(grep -oE -- '--llm-batch-collect [A-Za-z0-9_-]+' 25-batch-api/submit.log | head -1 | awk '{print $2}')
    note "batch id $BATCH_ID; collecting for up to ${BATCH_WAIT}s"
    DEADLINE=$(( $(date +%s) + BATCH_WAIT ))
    BATCH_RC=2
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
      MODEL="$E2E_BATCH_MODEL" markitai --llm-batch-collect "$BATCH_ID" -o 25-batch-api/output/ \
        >25-batch-api/collect.log 2>&1
      BATCH_RC=$?
      [ "$BATCH_RC" -ne 2 ] && break
      sleep 30
    done
    if [ "$BATCH_RC" -eq 0 ]; then
      ok "--llm-batch-collect finished the job later"
    elif [ "$BATCH_RC" -eq 2 ]; then
      skip "the provider had not finished the batch within ${BATCH_WAIT}s (raise BATCH_WAIT); the collect command still works later"
    else
      bad "--llm-batch-collect failed (see collect.log)"
    fi
  else
    bad "--llm-batch failed (see submit.log)"
  fi
  if [ "$BATCH_RC" -eq 0 ]; then
    BATCH_DONE=$(ls 25-batch-api/output/*.llm.md 2>/dev/null | wc -l | tr -d ' ')
    check "every document came back enhanced ($BATCH_DONE of 3)" test "$BATCH_DONE" -eq 3
    check "and the run says the work was billed at half price" \
      grep -qE '50%' 25-batch-api/submit.log 25-batch-api/collect.log
  fi
  file "submission" 25-batch-api/submit.log
  [ -f 25-batch-api/collect.log ] && file "collection" 25-batch-api/collect.log
fi

# ── Report ───────────────────────────────────────────────────────────────────
DURATION=$(( $(date +%s) - STARTED_EPOCH ))
COST=$(grep -rhoE '\$[0-9]+\.[0-9]+' --include='*.log' . 2>/dev/null \
  | tr -d '$' | awk '{t += $1} END {printf "$%.3f", t + 0}')
log META version "$VERSION"
log META started "$(date '+%Y-%m-%d %H:%M')"
log META duration "${DURATION}s"
log META model "$E2E_MODEL"
log META cost "${COST:-—}"
log META command "scripts/e2e_release_check.sh"

python3 "$REPO_ROOT/scripts/e2e_report.py" "$RESULTS" "$WORKDIR/report.html" \
  || printf '%sreport rendering failed%s\n' "$Y" "$N"

printf '\n%s── Summary%s\n' "$B" "$N"
printf '  %s%d passed%s, %s%d failed%s · %ss · %s spent\n' \
  "$G" "$PASS" "$N" "$([ "$FAIL" -gt 0 ] && printf '%s' "$R")" "$FAIL" "$N" \
  "$DURATION" "${COST:-\$0}"

if [ "$FAIL" -eq 0 ] && [ "$CLEANUP_ON_SUCCESS" = "1" ]; then
  cd "$REAL_HOME" || cd /
  rm -rf "$WORKDIR"
  printf '  %severything passed; artifacts removed (CLEANUP_ON_SUCCESS=1)%s\n' "$D" "$N"
else
  printf '\n  %sReport:%s %s/report.html\n' "$B" "$N" "$WORKDIR"
  printf '  %sone numbered directory per step beside it, with that step'"'"'s inputs, outputs and logs%s\n' "$D" "$N"
fi

[ "$FAIL" -eq 0 ]
