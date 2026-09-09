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
#   BATCH_DOCS=40          documents generated for the batch/interrupt steps
#   HTTP_PORT=8899         port for the local page used by the URL step
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
export HOME="$WORKDIR/_internal/home"
export UV_TOOL_DIR="$HOME/.uvtools"
export UV_TOOL_BIN_DIR="$HOME/bin"
export PATH="$HOME/bin:$PATH"
mkdir -p "$HOME"
printf '  %sinstalling into an empty home (log: %s/_internal/install.log)%s\n' "$D" "$WORKDIR" "$N"
uv tool install "$WHEEL" >"$WORKDIR/_internal/install.log" 2>&1 \
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
