// Updating a loopback server from the workbench (contract v1.24, owner picks 2026-10-25): entry E3 (an icon
// with a dot beside the sidebar toggle / before Docs and GitHub on a phone) shown only with `update`, the
// round-3 card (C1 inline confirm, P1 stepper and status row, six failures with what they left), the busy
// note, the held Convert and the reload after the restart.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { UpdateError, UpdateStatus, UpdateSummary } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { hasUnsaved, setUnsaved } from "./unsaved.ts";
import { clock, downloaded, entryLook, errorKey, failedStep, failureText, fraction, HOLDING, notesFor, pollDelay, runningVersion, shownError, statusRow, stepOf } from "./update.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");

const status = (over: Partial<UpdateStatus> = {}): UpdateStatus => ({
  state: "idle",
  current: "2.1.0",
  latest: "2.2.0",
  newer: true,
  notes_url: "https://github.com/Ynewtime/markitai/releases/tag/v2.2.0",
  lead: "Sharper PDF tables.",
  lead_zh: null,
  method: "setup",
  command: "markitai update",
  can_apply: true,
  target: null,
  progress: null,
  waiting_jobs: 0,
  error: null,
  updated_from: null,
  by: null,
  ...over,
});
const summary = (over: Partial<UpdateSummary> = {}): UpdateSummary => ({ current: "2.1.0", latest: "2.2.0", newer: true, can_apply: true, method: "setup", state: "idle", ...over });
const failure = (code: string, over: Partial<UpdateError> = {}): UpdateError => ({ code, reason: "", rolled_back: false, ...over });

test("the entry shows only what the server reports: nothing without update, a dot, a spinner, red", () => {
  assert.equal(entryLook(undefined, null, null), null, "an exposed server or a user token: no entry at all");
  assert.equal(entryLook(summary({ newer: false }), null, null), null, "up to date: nothing to say");
  assert.equal(entryLook(summary(), null, null), "available");
  for (const state of ["downloading", "verifying", "waiting", "installing", "restarting"] as const) {
    assert.equal(entryLook(summary(), status({ state }), null), "busy", state);
  }
  const failed = status({ state: "failed", error: failure("checksum") });
  assert.equal(entryLook(summary(), failed, null), "failed");
  // "OK" on a failure hides it in this browser; the release is still there.
  assert.equal(entryLook(summary(), failed, errorKey(failed.error!)), "available");
  // An update the last start undid shows on an idle server too.
  const undone = status({ error: failure("start_failed", { rolled_back: true, version: "2.2.0" }) });
  assert.equal(shownError(undone, null)?.code, "start_failed");
  assert.equal(entryLook(summary({ newer: false }), { ...undone, newer: false }, null), "failed");
});

test("the stepper and the status row follow the state; the value is the download, SHA-256 or the wait", () => {
  assert.deepEqual(["downloading", "verifying", "waiting", "installing", "restarting", "idle"].map((state) => stepOf(state as never)), [0, 1, 2, 2, 3, -1]);
  const progress = { done: 12_100_000, total: 31_200_000 };
  assert.equal(downloaded(progress), "12.1 / 31.2 MB");
  assert.equal(downloaded({ done: 12_100_000, total: null }), "12.1 MB");
  assert.equal(Math.round((fraction(status({ progress })) ?? 0) * 100), 39);
  assert.equal(fraction(status({ progress: { done: 5, total: null } })), null);
  assert.deepEqual(statusRow(zh, status({ state: "downloading", progress }), 0), { text: "正在下载", value: "12.1 / 31.2 MB" });
  assert.deepEqual(statusRow(zh, status({ state: "verifying" }), 0), { text: "正在校验", value: "SHA-256" });
  assert.deepEqual(statusRow(zh, status({ state: "waiting", waiting_jobs: 2 }), 42.7), { text: "等待 2 个转换完成", value: "0:42" });
  assert.deepEqual(statusRow(en, status({ state: "waiting", waiting_jobs: 1 }), 3), { text: "Waiting for 1 conversion", value: "0:03" });
  assert.deepEqual(statusRow(zh, status({ state: "installing" }), 0), { text: "正在安装", value: "" });
  assert.equal(clock(3723), "1:02:03");
  // New conversions are held from waiting on.
  assert.deepEqual([...HOLDING], ["waiting", "installing", "restarting"]);
});

test("the six failures say what happened and what they left", () => {
  const cases: [UpdateError, string, string, number][] = [
    [failure("offline", { reason: "Can't reach github.com: connection timed out" }), "无法连接 github.com", "未做改动", 0],
    [failure("checksum"), "校验失败，已删除下载", "未做改动", 1],
    [failure("disk_full", { reason: "Not enough disk space: /home/u/.local/bin needs 32 MB" }), "磁盘空间不足（需要 32 MB）", "未做改动", 0],
    [failure("file_busy", { reason: "Can't replace markitai-mcp.exe: another program is using it", rolled_back: true }), "markitai-mcp.exe 正被占用", "已恢复，仍是 2.1.0", 2],
    [failure("not_writable", { reason: "Can't write to ~/.local/bin: permission denied" }), "无法写入 ~/.local/bin", "未做改动", 0],
    [failure("start_failed", { rolled_back: true, version: "2.2.0" }), "2.2.0 无法启动", "已回到 2.1.0", 3],
    [failure("cancelled"), "已取消", "未做改动", -1],
  ];
  for (const [error, title, left, step] of cases) {
    assert.deepEqual(failureText(zh, error, "2.1.0", "2.2.0"), { title, left }, error.code);
    assert.equal(failedStep(error), step, error.code);
  }
  // A new version that did not pass its dry start stopped at verifying, and nothing changed.
  assert.equal(failedStep(failure("start_failed")), 1);
  assert.equal(failureText(en, failure("offline", { reason: "odd" }), "2.1.0", "2.2.0").title, "Can't reach the download server");
});

test("the page asks often only while it matters", () => {
  assert.equal(pollDelay("idle", false), 30_000);
  assert.equal(pollDelay(null, false), 30_000);
  // An open card follows another tab's start or failure.
  assert.equal(pollDelay("idle", true), 3000);
  assert.equal(pollDelay("failed", true), 3000);
  assert.equal(pollDelay("downloading", false), 3000);
  assert.equal(pollDelay("downloading", true), 1000);
  assert.equal(pollDelay("restarting", false), 1000);
  assert.equal(notesFor("https://github.com/Ynewtime/markitai/", "2.2.0"), "https://github.com/Ynewtime/markitai/releases/tag/v2.2.0");
});

test("a reload after the restart waits for unsaved edits", () => {
  const empty = { querySelector: () => null };
  assert.equal(hasUnsaved(empty), false);
  setUnsaved("config", true);
  assert.equal(hasUnsaved(empty), true);
  setUnsaved("config", false);
  assert.equal(hasUnsaved({ querySelector: (selector: string) => (selector === ".veil:not(.upd-veil)" ? ({} as Element) : null) }), true);
  assert.match(read("../components/settings-config.tsx"), /setUnsaved\("config", changed\)/);
});

test("every language has the update's words, Chinese with full-width punctuation", () => {
  for (const dict of [en, zh, ja]) {
    assert.equal(dict.updSteps.length, 4);
    assert.ok(dict.updConfirmNote(2) && dict.updWaiting(2) && dict.updDone("2.2.0"));
  }
  assert.equal(zh.updConfirmNote(2), "服务将重启，2 个转换会先完成");
  assert.equal(zh.updCopyCommand("markitai update"), "复制命令：markitai update");
  assert.equal(zh.updErrDisk("32 MB"), "磁盘空间不足（需要 32 MB）");
  for (const value of Object.entries(zh).filter(([key]) => key.startsWith("upd"))) {
    const texts = typeof value[1] === "function" ? [(value[1] as (...a: unknown[]) => string)("2.2.0", "x")] : Array.isArray(value[1]) ? value[1] : [value[1]];
    for (const text of texts) assert.ok(!/[一-鿿][,:;?!()]|[,:;?!(][一-鿿]/.test(String(text)), `${value[0]}: ${text}`);
  }
});

test("E3 sits beside the sidebar toggle and before the phone's links; the card is a flyout or a sheet", () => {
  const shell = read("../components/shell.tsx");
  assert.ok(shell.indexOf('updateEntry?.("sidebar")') < shell.indexOf('class="rail-toggle"'), "before the sidebar toggle");
  const links = shell.slice(shell.indexOf('<span class="phone-links">'));
  assert.ok(links.indexOf('updateEntry?.("phone")') < links.indexOf("href={docsUrl}"), "first of the phone's links");
  const card = read("../components/update.tsx");
  assert.match(card, /class="menu upd-pop"/);
  assert.match(card, /dialog opts-dialog upd-sheet/);
  // No settings row and no tab dot (owner, E3).
  assert.ok(!/upd/.test(read("../components/settings-page.tsx")));
  assert.ok(!/upd/.test(shell.slice(shell.indexOf('<nav class="tabbar"'))));
  const app = read("../app.tsx");
  assert.match(app, /held=\{update\.holding\}/, "Convert is held while the server updates");
  assert.match(app, /updateEntry=\{\(place\) => <UpdateEntry/);
  // The status row's value is one step smaller than its text, the same sans with tabular figures.
  const css = read("../styles/app.css");
  const num = css.slice(css.indexOf(".upd-num {"), css.indexOf("}", css.indexOf(".upd-num {")));
  assert.match(num, /font-size: var\(--text-xs\)/);
  assert.match(num, /tabular-nums/);
  assert.ok(!/font-family/.test(num));
  // Phone controls are 28 px with 11 px text.
  assert.match(css, /\.upd-sheet \.btn \{\s*height: 28px;\s*padding: 2px 10px 0;\s*font-size: var\(--text-xs\);/);
});

test("the card's \"now\" stays the running version until the page has reconnected to the new build", () => {
  // The server names the installed version while it is still installing or restarting.
  for (const state of ["downloading", "verifying", "waiting", "installing", "restarting"] as const) {
    assert.equal(runningVersion("2.1.0", status({ state, current: "2.2.0", target: "2.2.0" })), "2.1.0", state);
  }
  // Reconnected to the new build (or reloaded into it): the hook's running version is the new one.
  assert.equal(runningVersion("2.2.0", status({ current: "2.2.0" })), "2.2.0");
  // Before the page knows its own version, the server's.
  assert.equal(runningVersion(null, status()), "2.1.0");
  const hook = read("../hooks/use-update.ts");
  assert.equal(hook.match(/setReconnectedTo\(/g)?.length, 1, "only the reconnection changes the running version");
  assert.match(hook, /setReconnectedTo\(target\);\s*setReconnecting\(false\);/);
  assert.match(read("../components/update.tsx"), /t\.updCurrent\(current\)/);
  assert.match(read("../components/update.tsx"), /const current = runningVersion\(update\.running, status\);/);
});

test("on a phone the done notice's Release notes reads like the page's buttons, not spaced capitals", () => {
  const css = read("../styles/app.css");
  const phone = css.slice(css.indexOf("@media (max-width: 760px) {\n  /* a notice's action on a phone"));
  const rule = phone.slice(0, phone.indexOf("}"));
  assert.match(rule, /\.toast\.is-line \.note-action/);
  assert.match(rule, /text-transform: none;/);
  assert.match(rule, /letter-spacing: 0;/);
  assert.match(rule, /font-family: inherit;/);
  // After the one-line notice's own offset, so it wins on a phone.
  assert.ok(css.indexOf(rule) > css.indexOf(".toast.is-line .note-action {\n  position: relative;"));
});
