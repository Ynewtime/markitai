#!/usr/bin/env python3
"""Render the end-to-end check's results as a page someone will actually read.

Reads the TSV log ``e2e_release_check.sh`` writes as it goes and produces a
self-contained ``report.html`` next to the run's artifacts. Every claim links
to the file that backs it, and conversions are shown inline: a reader judging
whether markitai is ready to ship needs to see what it produced, not only
that a check passed.

Record types, tab separated:

    META      key      value
    STEP      id       title            subtitle
    CHECK     id       ok|fail|skip     text
    NOTE      id       text
    EVIDENCE  id       label            relative-path    [inline]

``inline`` renders the file's first lines in the page instead of only linking
to it.
"""

from __future__ import annotations

import html
import sys
from dataclasses import dataclass, field
from pathlib import Path

INLINE_LINES = 26
INLINE_CHARS = 2600


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


def preview(root: Path, rel: str) -> str | None:
    target = root / rel
    if not target.is_file():
        return None
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    clipped = "\n".join(lines[:INLINE_LINES])
    if len(clipped) > INLINE_CHARS:
        clipped = clipped[:INLINE_CHARS]
    if len(lines) > INLINE_LINES or len(clipped) < len(text):
        clipped += "\n…"
    return clipped or "(empty)"


STYLE = """
:root {
  color-scheme: light dark;
  --bg: #fbfbfa; --panel: #ffffff; --ink: #1b1b1a; --muted: #6b6b68;
  --line: #e6e5e2; --ok: #2f7d51; --fail: #b3402f; --skip: #9a7b2f;
  --accent: #2f5d7d; --code-bg: #f6f6f4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17171a; --panel: #1e1e22; --ink: #ececea; --muted: #9a9a97;
    --line: #2e2e33; --ok: #6cc08a; --fail: #e08573; --skip: #d6b45f;
    --accent: #8ab6d6; --code-bg: #232329;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", "Helvetica Neue",
        "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 56px 24px 96px; }
header { margin-bottom: 40px; }
h1 { font-size: 25px; letter-spacing: -0.01em; margin: 0 0 6px; font-weight: 620; }
.sub { color: var(--muted); margin: 0 0 24px; }
.verdict {
  display: inline-flex; align-items: baseline; gap: 10px;
  padding: 10px 16px; border-radius: 10px; font-weight: 620; font-size: 17px;
  border: 1px solid transparent;
}
.verdict.ok { background: color-mix(in srgb, var(--ok) 12%, transparent);
  color: var(--ok); border-color: color-mix(in srgb, var(--ok) 30%, transparent); }
.verdict.fail { background: color-mix(in srgb, var(--fail) 12%, transparent);
  color: var(--fail); border-color: color-mix(in srgb, var(--fail) 30%, transparent); }
.verdict small { font-weight: 450; color: var(--muted); font-size: 14px; }
.meta {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; margin-top: 28px;
}
.meta div { background: var(--panel); padding: 12px 14px; }
.meta dt { color: var(--muted); font-size: 12px; letter-spacing: 0.03em;
  text-transform: uppercase; margin: 0 0 3px; }
.meta dd { margin: 0; font-variant-numeric: tabular-nums; }
section {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 22px 24px; margin-bottom: 16px;
}
section.fail { border-color: color-mix(in srgb, var(--fail) 45%, var(--line)); }
.head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }
.num { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 13px;
  min-width: 20px; }
h2 { font-size: 16.5px; margin: 0; font-weight: 600; letter-spacing: -0.005em; }
.why { color: var(--muted); margin: 0 0 16px 32px; font-size: 14px; }
ul.checks { list-style: none; padding: 0; margin: 0 0 0 32px; }
ul.checks li { display: flex; gap: 10px; padding: 4px 0; align-items: baseline; }
.mark { font-weight: 700; width: 14px; flex: none; }
.ok .mark, .mark.ok { color: var(--ok); }
.fail .mark, .mark.fail { color: var(--fail); }
.skip .mark, .mark.skip { color: var(--skip); }
li.skip span:last-child { color: var(--muted); }
.notes { margin: 14px 0 0 32px; color: var(--muted); font-size: 13.5px; }
.notes div { padding: 2px 0; }
figure { margin: 18px 0 0 32px; }
figcaption { font-size: 12.5px; color: var(--muted); margin-bottom: 6px;
  display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
figcaption a { color: var(--accent); text-decoration: none; }
figcaption a:hover { text-decoration: underline; }
pre {
  margin: 0; background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; padding: 12px 14px; overflow-x: auto;
  font: 12.5px/1.55 ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
.files { margin: 16px 0 0 32px; font-size: 13.5px; }
.files a { color: var(--accent); text-decoration: none; }
.files a:hover { text-decoration: underline; }
.files span { color: var(--muted); }
footer { margin-top: 40px; color: var(--muted); font-size: 13.5px; }
footer code { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 5px; padding: 1px 6px;
  font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; }
"""

MARKS = {"ok": "✓", "fail": "✗", "skip": "—"}


def render(meta: dict[str, str], steps: list[Step], root: Path) -> str:
    passed = sum(1 for s in steps for c in s.checks if c.status == "ok")
    failed = sum(1 for s in steps for c in s.checks if c.status == "fail")
    skipped = sum(1 for s in steps for c in s.checks if c.status == "skip")
    verdict = "fail" if failed else "ok"
    e = html.escape

    rows = [
        ("Version", meta.get("version", "—")),
        ("Run", meta.get("started", "—")),
        ("Duration", meta.get("duration", "—")),
        ("Model", meta.get("model", "—")),
        ("LLM spend", meta.get("cost", "—")),
        ("Checks", f"{passed} passed · {failed} failed · {skipped} skipped"),
    ]

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>markitai end-to-end check — {e(meta.get('version', ''))}</title>",
        f"<style>{STYLE}</style></head><body><div class='wrap'>",
        "<header>",
        "<h1>markitai — end-to-end release check</h1>",
        (
            "<p class='sub'>The installed product on a machine that has never "
            "seen it, driven only through the public command line, with a real "
            "model.</p>"
        ),
        f"<div class='verdict {verdict}'>",
        ("Ready to ship" if verdict == "ok" else "Not ready"),
        f"<small>{passed} of {passed + failed} checks passed</small></div>",
        "<dl class='meta'>",
    ]
    for label, value in rows:
        out.append(f"<div><dt>{e(label)}</dt><dd>{e(value)}</dd></div>")
    out.append("</dl></header>")

    for number, step in enumerate(steps, start=1):
        out.append(f"<section class='{step.status}'>")
        out.append(
            f"<div class='head'><span class='num'>{number:02d}</span>"
            f"<h2>{e(step.title)}</h2></div>"
        )
        out.append(f"<p class='why'>{e(step.subtitle)}</p>")
        if step.checks:
            out.append("<ul class='checks'>")
            for check in step.checks:
                out.append(
                    f"<li class='{check.status}'>"
                    f"<span class='mark {check.status}'>{MARKS[check.status]}</span>"
                    f"<span>{e(check.text)}</span></li>"
                )
            out.append("</ul>")
        if step.notes:
            out.append("<div class='notes'>")
            out.extend(f"<div>{e(note)}</div>" for note in step.notes)
            out.append("</div>")

        linked = []
        for item in step.evidence:
            if item.inline:
                body = preview(root, item.path)
                if body is None:
                    continue
                out.append(
                    "<figure><figcaption><span>"
                    f"{e(item.label)}</span>"
                    f"<a href='{e(item.path)}'>{e(item.path)}</a>"
                    f"</figcaption><pre>{e(body)}</pre></figure>"
                )
            else:
                linked.append(item)
        if linked:
            out.append("<div class='files'>")
            out.append(
                " · ".join(f"<a href='{e(i.path)}'>{e(i.label)}</a>" for i in linked)
            )
            out.append("</div>")
        out.append("</section>")

    out.append(
        "<footer>Artifacts live beside this file, one directory per step. "
        f"Re-run with <code>{e(meta.get('command', 'scripts/e2e_release_check.sh'))}"
        "</code>; set <code>CLEANUP_ON_SUCCESS=1</code> to discard them on a "
        "clean pass.</footer>"
    )
    out.append("</div></body></html>")
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
