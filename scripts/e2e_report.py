#!/usr/bin/env python3
"""Render the end-to-end check's results as a page someone will actually read.

Reads the TSV log ``e2e_release_check.sh`` writes as it goes and produces a
self-contained ``report.html`` next to the run's artifacts. Every claim links
to the file that backs it, and conversions are shown inline: a reader judging
whether markitai is ready to ship needs to see what it produced, not only
that a check passed. Failures are surfaced twice — in a callout at the top and
on the step itself — and a sticky step bar gives the run's shape at a glance
and a way to jump.

Record types, tab separated:

    META      key      value
    STEP      id       title            subtitle
    CHECK     id       ok|fail|skip     text
    NOTE      id       text
    EVIDENCE  id       label            relative-path    [inline]

``inline`` renders the file into the page instead of only linking to it:
terminal noise (ANSI colors, progress-bar overwrites) is cleaned away, files
up to a screenful render unfolded, and longer ones fold open on demand so a
reader can read the whole conversion without leaving the report.
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# How much of an inline evidence file is embedded in the page. Up to OPEN_*
# renders unfolded; everything embeds up to EMBED_* inside a fold, and the
# full file is one click away past that.
OPEN_LINES = 26
OPEN_CHARS = 2600
EMBED_LINES = 200
EMBED_CHARS = 20_000

# Colors, cursor moves, OSC sequences — what the terminal drew, not what the
# tool said.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


@dataclass
class Check:
    status: str
    text: str


@dataclass
class Evidence:
    label: str
    path: str
    inline: bool


@dataclass
class Snippet:
    text: str
    lines: int  # lines in the whole file
    shown: int  # lines embedded in the page
    clipped: bool


@dataclass
class Step:
    ident: str
    title: str
    subtitle: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if not self.checks or all(c.status == "skip" for c in self.checks):
            return "skip"
        return "ok"


def parse(log: Path) -> tuple[dict[str, str], list[Step]]:
    meta: dict[str, str] = {}
    steps: list[Step] = []
    index: dict[str, Step] = {}

    for raw in log.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        kind = parts[0]
        if kind == "META" and len(parts) >= 3:
            meta[parts[1]] = parts[2]
        elif kind == "STEP" and len(parts) >= 4:
            step = Step(parts[1], parts[2], parts[3])
            steps.append(step)
            index[step.ident] = step
        elif kind == "CHECK" and len(parts) >= 4 and parts[1] in index:
            index[parts[1]].checks.append(Check(parts[2], parts[3]))
        elif kind == "NOTE" and len(parts) >= 3 and parts[1] in index:
            index[parts[1]].notes.append(parts[2])
        elif kind == "EVIDENCE" and len(parts) >= 4 and parts[1] in index:
            index[parts[1]].evidence.append(
                Evidence(parts[2], parts[3], len(parts) > 4 and parts[4] == "inline")
            )
    return meta, steps


def anchor(ident: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", ident)


def plain(text: str) -> str:
    """Terminal output as a reader sees it: colors gone, each progress bar
    reduced to its last frame."""
    text = ANSI.sub("", text)
    cleaned = (
        seg.rsplit("\r", 1)[-1] if "\r" in seg else seg for seg in text.split("\n")
    )
    return "\n".join(seg.rstrip() for seg in cleaned)


def embed(root: Path, rel: str) -> Snippet | None:
    target = root / rel
    if not target.is_file():
        return None
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = plain(raw).split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the trailing newline is not a line of output
    total = len(lines)
    body = "\n".join(lines[:EMBED_LINES])
    clipped = total > EMBED_LINES or len(body) > EMBED_CHARS
    if len(body) > EMBED_CHARS:
        body = body[:EMBED_CHARS]
        shown = body.count("\n") + 1
    else:
        shown = min(total, EMBED_LINES)
    if clipped:
        body += "\n…"
    return Snippet(body or "(empty)", total, shown, clipped)


ICON_FILE = (
    "<svg width='11' height='11' viewBox='0 0 16 16' fill='none' aria-hidden='true'>"
    "<path d='M4 1.5h5.2L13 5.3v9.2H4z' stroke='currentColor' stroke-width='1.3' "
    "stroke-linejoin='round'/><path d='M9 1.8V5.5h3.7' stroke='currentColor' "
    "stroke-width='1.3' stroke-linejoin='round'/></svg>"
)

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #faf9f7; --panel: #ffffff; --ink: #1d1d1b; --muted: #71716c;
  --line: #e8e6e1; --line-strong: #d9d6cf;
  --ok: #22794c; --ok-bg: #e9f4ed;
  --fail: #b23a2b; --fail-bg: #f9e9e6;
  --skip: #92713a; --skip-bg: #f5efdf;
  --accent: #33618c; --code-bg: #f5f4f0;
  --shadow: 0 1px 2px rgb(28 28 26 / 4%), 0 10px 28px -20px rgb(28 28 26 / 25%);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #151517; --panel: #1d1d20; --ink: #ececea; --muted: #9c9c96;
    --line: #2b2b30; --line-strong: #3b3b41;
    --ok: #74c190; --ok-bg: #1e2d25;
    --fail: #e28573; --fail-bg: #32201c;
    --skip: #d4b267; --skip-bg: #2e2819;
    --accent: #8fb8da; --code-bg: #242429;
    --shadow: none;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", "Helvetica Neue",
        "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
::selection { background: color-mix(in srgb, var(--accent) 25%, transparent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ─ sticky step bar ─────────────────────────────────────────────────────── */
.bar {
  position: sticky; top: 0; z-index: 20;
  background: color-mix(in srgb, var(--bg) 80%, transparent);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
}
.bar-in { max-width: 1140px; margin: 0 auto; padding: 8px 20px;
  display: flex; align-items: center; gap: 18px; }
.brand { font-weight: 650; font-size: 13.5px; color: var(--ink);
  text-decoration: none; white-space: nowrap; }
.brand .v { color: var(--muted); font-weight: 480; margin-left: 7px;
  font-variant-numeric: tabular-nums; }
.dots { display: flex; align-items: center; gap: 4px; list-style: none;
  margin: 0; padding: 0; overflow-x: auto; scrollbar-width: none; flex: 1; }
.dots::-webkit-scrollbar { display: none; }
.dots li { flex: none; }
.dots a { display: flex; align-items: baseline; gap: 6px; padding: 3px 9px;
  border-radius: 8px; text-decoration: none; color: var(--muted);
  border: 1px solid transparent; font-size: 12px; line-height: 1.5; }
.dots a .n { font: 600 11.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums; }
.dots a .t { max-width: 190px; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
@media (max-width: 999px) { .dots a .t { display: none; } }
.dots a.ok .n { color: var(--ok); }
.dots a.fail { color: var(--fail); background: var(--fail-bg);
  border-color: color-mix(in srgb, var(--fail) 40%, transparent); }
.dots a:hover, .dots a.here { color: var(--ink); background: var(--panel);
  border-color: var(--line); }
.dots a.here { border-color: var(--line-strong); box-shadow: var(--shadow); }
.ghost { flex: none; font: inherit; font-size: 12px; font-weight: 500;
  color: var(--muted); background: var(--panel); border: 1px solid var(--line-strong);
  border-radius: 8px; padding: 4px 10px; cursor: pointer; }
.ghost:hover { color: var(--ink); }

/* ─ page ────────────────────────────────────────────────────────────────── */
.wrap { max-width: 900px; margin: 0 auto; padding: 48px 24px 40px; }
.hero { margin-bottom: 30px; }
h1 { font-size: 27px; letter-spacing: -0.015em; margin: 0 0 8px; font-weight: 680; }
.sub { color: var(--muted); margin: 0; max-width: 58ch; }
.verdict { display: inline-flex; align-items: center; gap: 12px; margin-top: 24px;
  padding: 13px 18px 13px 14px; border-radius: 12px; font-weight: 650; font-size: 16px;
  border: 1px solid transparent; }
.verdict.ok { background: var(--ok-bg); color: var(--ok);
  border-color: color-mix(in srgb, var(--ok) 32%, transparent); }
.verdict.fail { background: var(--fail-bg); color: var(--fail);
  border-color: color-mix(in srgb, var(--fail) 40%, transparent); }
.vmark { width: 27px; height: 27px; border-radius: 50%; display: grid;
  place-items: center; font-size: 13px; color: #fff; background: var(--ok); flex: none; }
.verdict.fail .vmark { background: var(--fail); }
.verdict small { display: block; font-weight: 450; font-size: 12.5px;
  color: var(--muted); margin-top: 1px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 26px 0 0; }
.stats > div { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 11px 14px; box-shadow: var(--shadow); min-width: 0; }
.stats dt { color: var(--muted); font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; margin: 0 0 3px; }
.stats dd { margin: 0; font-size: 14.5px; font-weight: 560;
  font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }

/* failure callout */
.failbox { background: var(--fail-bg); border: 1px solid
  color-mix(in srgb, var(--fail) 38%, transparent);
  border-radius: 12px; padding: 14px 18px; margin: 26px 0 2px; }
.failbox .ft { margin: 0 0 6px; color: var(--fail); font-weight: 650; font-size: 13.5px; }
.failbox ul { margin: 0; padding: 0; list-style: none; }
.failbox li { font-size: 13.5px; padding: 3px 0; }
.failbox li a { font-weight: 600; }

/* steps */
section.step { background: var(--panel); border: 1px solid var(--line);
  border-radius: 14px; padding: 22px 24px; margin: 16px 0; box-shadow: var(--shadow);
  scroll-margin-top: 64px; }
section.step.fail { border-color: color-mix(in srgb, var(--fail) 45%, var(--line)); }
.head { display: flex; align-items: flex-start; gap: 12px; }
.num { font: 600 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--muted); background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 7px; padding: 6px 7px; margin-top: 1px;
  font-variant-numeric: tabular-nums; }
.ht { flex: 1; min-width: 0; }
h2 { font-size: 16.5px; margin: 0; font-weight: 640; letter-spacing: -0.005em; }
.why { color: var(--muted); font-size: 13.5px; margin: 3px 0 0; }
.pill { flex: none; font-size: 12px; font-weight: 600; padding: 4px 10px;
  border-radius: 99px; border: 1px solid var(--line); color: var(--muted);
  margin-top: 1px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pill.ok { color: var(--ok); background: var(--ok-bg);
  border-color: color-mix(in srgb, var(--ok) 30%, transparent); }
.pill.fail { color: var(--fail); background: var(--fail-bg);
  border-color: color-mix(in srgb, var(--fail) 38%, transparent); }
.pill.skip { color: var(--skip); background: var(--skip-bg);
  border-color: color-mix(in srgb, var(--skip) 32%, transparent); }

ul.checks { list-style: none; padding: 0; margin: 16px 0 0; }
ul.checks li { display: flex; gap: 10px; padding: 4.5px 10px 4.5px 8px;
  align-items: baseline; border-radius: 8px; font-size: 14.5px; }
ul.checks li.fail { background: var(--fail-bg); }
.mark { font-weight: 700; width: 13px; flex: none; text-align: center; }
.mark.ok { color: var(--ok); }
.mark.fail { color: var(--fail); }
.mark.skip { color: var(--skip); }
li.skip span:last-child { color: var(--muted); }

.notes { margin: 13px 0 0; display: grid; gap: 5px; }
.notes div { font: 12.5px/1.55 ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  color: var(--muted); padding: 1px 0 1px 11px; border-left: 2px solid var(--line-strong);
  overflow-wrap: anywhere; }

/* evidence */
figure.ev { margin: 17px 0 0; }
figcaption { display: flex; align-items: center; gap: 10px; font-size: 12.5px;
  margin-bottom: 6px; min-width: 0; }
figcaption .lbl { font-weight: 600; }
figcaption a { margin-left: auto; color: var(--accent); text-decoration: none;
  font: 11.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
figcaption a:hover { text-decoration: underline; }
.copy { flex: none; font: inherit; font-size: 11px; font-weight: 550;
  color: var(--muted); background: var(--panel); border: 1px solid var(--line-strong);
  border-radius: 6px; padding: 2px 8px; cursor: pointer; }
.copy:hover { color: var(--ink);
  border-color: color-mix(in srgb, var(--accent) 45%, var(--line-strong)); }
.copy.done { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 45%, transparent); }

details.fold summary { list-style: none; cursor: pointer; display: flex;
  align-items: center; gap: 8px; font-size: 12px; color: var(--muted);
  padding: 6px 12px; background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; user-select: none; font-variant-numeric: tabular-nums; }
details.fold summary::-webkit-details-marker { display: none; }
details.fold summary::before { content: "▸"; font-size: 10px;
  transition: transform 0.15s ease; }
details.fold[open] summary::before { transform: rotate(90deg); }
details.fold summary:hover { color: var(--ink); }
details.fold .act { margin-left: auto; font-size: 11px; }
details.fold .act::after { content: "expand"; }
details.fold[open] .act::after { content: "fold"; }
details.fold[open] summary { border-radius: 8px 8px 0 0; }
details.fold pre { border-top: 0; border-radius: 0 0 8px 8px; }

pre { margin: 0; background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; padding: 12px 14px; overflow-x: auto;
  font: 12.5px/1.55 ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace; }
.clip { font-size: 12px; color: var(--muted); margin: 6px 0 0; }

.files { margin: 15px 0 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.flabel { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.08em; margin-right: 2px; }
.files a { display: inline-flex; align-items: center; gap: 7px; color: var(--accent);
  text-decoration: none; border: 1px solid var(--line); background: var(--code-bg);
  padding: 4px 10px; border-radius: 8px; font-size: 12.5px; }
.files a:hover { border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }
.files a svg { flex: none; opacity: 0.65; }
.files .miss { color: var(--fail); font-size: 11px; font-weight: 600; }

footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 13px; }
footer code { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 5px; padding: 1px 6px;
  font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }

@media (max-width: 640px) {
  .wrap { padding: 28px 14px 40px; }
  .bar-in { padding: 8px 12px; gap: 10px; }
  h1 { font-size: 22px; }
  section.step { padding: 16px 14px; border-radius: 12px; }
  .head { flex-wrap: wrap; }
  figcaption { flex-wrap: wrap; }
  figcaption a { max-width: 100%; }
}
@media print {
  .bar, .copy, .ghost { display: none; }
  body { background: #fff; }
  section.step, figure.ev, ul.checks li, .stats > div, .failbox {
    box-shadow: none; break-inside: avoid; }
}
"""

SCRIPT = """
(function () {
  'use strict';
  var all = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  // copy buttons
  all('.copy').forEach(function (b) {
    b.addEventListener('click', function () {
      var figure = b.closest('figure');
      var pre = figure ? figure.querySelector('pre') : null;
      var text = pre ? pre.textContent : '';
      var done = function () {
        b.textContent = 'copied';
        b.classList.add('done');
        setTimeout(function () {
          b.textContent = 'copy';
          b.classList.remove('done');
        }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(); });
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        ta.remove();
      }
    });
  });

  // which step is on screen
  var dots = all('.dots a');
  var byId = {};
  dots.forEach(function (a) { byId[a.hash.slice(1)] = a; });
  if ('IntersectionObserver' in window) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        dots.forEach(function (a) { a.classList.remove('here'); });
        var dot = byId[en.target.id];
        if (dot) dot.classList.add('here');
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    Object.keys(byId).forEach(function (id) {
      var s = document.getElementById(id);
      if (s) spy.observe(s);
    });
  }
  // a short final step can end above the spy band; the bottom of the page
  // still belongs to it
  window.addEventListener('scroll', function () {
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 24) {
      dots.forEach(function (a) { a.classList.remove('here'); });
      if (dots.length) dots[dots.length - 1].classList.add('here');
    }
  }, { passive: true });

  // evidence folds
  var folds = all('details.fold');
  if (folds.length) {
    var toggle = document.getElementById('evtoggle');
    if (toggle) {
      toggle.hidden = false;
      toggle.addEventListener('click', function () {
        var open = folds.some(function (d) { return !d.open; });
        folds.forEach(function (d) { d.open = open; });
        toggle.textContent = open ? 'collapse evidence' : 'expand evidence';
      });
    }
    var wasOpen = [];
    window.addEventListener('beforeprint', function () {
      wasOpen = folds.map(function (d) { return d.open; });
      folds.forEach(function (d) { d.open = true; });
    });
    window.addEventListener('afterprint', function () {
      folds.forEach(function (d, i) { d.open = wasOpen[i]; });
    });
  }
})();
"""


def evidence_figure(item: Evidence, sn: Snippet) -> str:
    e = html.escape
    path = e(item.path)
    head = (
        "<figcaption>"
        f"<span class='lbl'>{e(item.label)}</span>"
        f"<a href='{path}' title='{path}'>{path}</a>"
        "<button class='copy' type='button'>copy</button>"
        "</figcaption>"
    )
    if not sn.clipped and sn.lines <= OPEN_LINES and len(sn.text) <= OPEN_CHARS:
        body = f"<pre>{e(sn.text)}</pre>"
    else:
        count = (
            f"first {sn.shown:,} of {sn.lines:,} lines"
            if sn.clipped
            else f"{sn.lines:,} lines"
        )
        body = (
            "<details class='fold'>"
            f"<summary>{e(count)}<span class='act'></span></summary>"
            f"<pre>{e(sn.text)}</pre>"
            "</details>"
        )
    clip = (
        ""
        if not sn.clipped
        else "<p class='clip'>the rest lives in the file beside this report</p>"
    )
    return f"<figure class='ev'>{head}{body}{clip}</figure>"


def step_pill(step: Step) -> tuple[str, str]:
    n = len(step.checks)
    nfail = sum(1 for c in step.checks if c.status == "fail")
    nok = sum(1 for c in step.checks if c.status == "ok")
    if nfail:
        return "fail", f"{nfail} of {n} failed"
    if n == 0:
        return "none", "no checks"
    if nok == 0:
        return "skip", "skipped"
    return "ok", f"{nok} of {n} passed"


def render(meta: dict[str, str], steps: list[Step], root: Path) -> str:
    e = html.escape
    passed = sum(1 for s in steps for c in s.checks if c.status == "ok")
    failed = sum(1 for s in steps for c in s.checks if c.status == "fail")
    skipped = sum(1 for s in steps for c in s.checks if c.status == "skip")
    total = passed + failed
    verdict = "fail" if failed else "ok"
    version = meta.get("version", "—")

    counts = [f"{passed} of {total} checks passed"]
    if failed:
        counts.append(f"{failed} failed")
    if skipped:
        counts.append(f"{skipped} skipped")

    stats = [
        ("Version", version),
        ("Started", meta.get("started", "—")),
        ("Duration", meta.get("duration", "—")),
        ("Model", meta.get("model", "—")),
        ("LLM spend", meta.get("cost", "—")),
    ]

    dots = "".join(
        f"<li><a class='{step.status}' href='#{e(anchor(step.ident))}' "
        f"title='{e(f'{number:02d} · {step.title}')}'>"
        f"<span class='n'>{number:02d}</span>"
        f"<span class='t'>{e(step.title)}</span></a></li>"
        for number, step in enumerate(steps, start=1)
    )

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>markitai {e(version)} — end-to-end check</title>",
        f"<style>{STYLE}</style></head><body>",
        (
            f"<nav class='bar'><div class='bar-in'>"
            f"<a class='brand' href='#top'>markitai"
            f"<span class='v'>{e(version)}</span></a>"
            f"<ol class='dots'>{dots}</ol>"
            "<button class='ghost' id='evtoggle' type='button' hidden>"
            "expand evidence</button></div></nav>"
        ),
        "<div class='wrap'>",
        "<header class='hero' id='top'>",
        "<h1>markitai — end-to-end release check</h1>",
        (
            "<p class='sub'>The installed product on a machine that has never "
            "seen it, driven only through the public command line, with a real "
            "model.</p>"
        ),
        f"<div class='verdict {verdict}'>",
        f"<span class='vmark'>{'✗' if failed else '✓'}</span>",
        "<span>" + ("Not ready" if failed else "Ready to ship") + "</span>",
        f"<small>{e(' · '.join(counts))}</small></div>",
        "<dl class='stats'>",
    ]
    for label, value in stats:
        out.append(f"<div><dt>{e(label)}</dt><dd>{e(value)}</dd></div>")
    out.append("</dl></header>")

    if failed:
        items = "".join(
            f"<li><a href='#{e(anchor(s.ident))}'>{e(s.title)}</a> — {e(c.text)}</li>"
            for s in steps
            for c in s.checks
            if c.status == "fail"
        )
        out.append(
            f"<div class='failbox'><p class='ft'>✗ {failed} check"
            f"{'s' if failed != 1 else ''} failed</p><ul>{items}</ul></div>"
        )

    for number, step in enumerate(steps, start=1):
        pill_cls, pill_text = step_pill(step)
        out.append(
            f"<section class='step {step.status}' id='{e(anchor(step.ident))}'>"
            "<div class='head'>"
            f"<span class='num'>{number:02d}</span>"
            f"<div class='ht'><h2>{e(step.title)}</h2>"
            f"<p class='why'>{e(step.subtitle)}</p></div>"
            f"<span class='pill {pill_cls}'>{e(pill_text)}</span>"
            "</div>"
        )
        if step.checks:
            out.append("<ul class='checks'>")
            marks = {"ok": "✓", "fail": "✗", "skip": "—"}
            for check in step.checks:
                out.append(
                    f"<li class='{check.status}'>"
                    f"<span class='mark {check.status}'>{marks[check.status]}</span>"
                    f"<span>{e(check.text)}</span></li>"
                )
            out.append("</ul>")
        if step.notes:
            out.append("<div class='notes'>")
            out.extend(f"<div>{e(note)}</div>" for note in step.notes)
            out.append("</div>")

        linked = []
        for item in step.evidence:
            if not item.inline:
                linked.append(item)
                continue
            sn = embed(root, item.path)
            if sn is None:
                out.append(
                    "<figure class='ev'><figcaption>"
                    f"<span class='lbl'>{e(item.label)}</span>"
                    f"<a href='{e(item.path)}'>{e(item.path)}</a></figcaption>"
                    "<p class='clip'>the file is gone from the artifacts "
                    "directory</p></figure>"
                )
                continue
            out.append(evidence_figure(item, sn))
        if linked:
            chips = "".join(
                f"<a href='{e(i.path)}'>{ICON_FILE}{e(i.label)}"
                + (
                    ""
                    if (root / i.path).is_file()
                    else "<span class='miss'>missing</span>"
                )
                + "</a>"
                for i in linked
            )
            out.append(
                f"<div class='files'><span class='flabel'>files</span>{chips}</div>"
            )
        out.append("</section>")

    command = e(meta.get("command", "scripts/e2e_release_check.sh"))
    out.append(
        "<footer>Artifacts live beside this file, one directory per step. "
        f"Re-run with <code>{command}</code>; set "
        "<code>CLEANUP_ON_SUCCESS=1</code> to discard them on a clean pass. "
        "Rendered from <code>_internal/results.tsv</code>.</footer>"
    )
    out.append(f"</div><script>{SCRIPT}</script></body></html>")
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: e2e_report.py <results.tsv> <report.html>", file=sys.stderr)
        return 2
    log, target = Path(sys.argv[1]), Path(sys.argv[2])
    meta, steps = parse(log)
    target.write_text(render(meta, steps, target.parent), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
