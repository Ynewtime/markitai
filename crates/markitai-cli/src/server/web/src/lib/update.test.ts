// Updating a loopback server from the workbench (contract v1.24, owner picks 2026-10-25): entry E3 (an icon
// with a dot beside the sidebar toggle / before Docs and GitHub on a phone) shown only with `update`, the
// round-3 card (C1 inline confirm, P1 stepper and status row, six failures with what they left), the busy
// note, the held Convert and the reload after the restart.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { ApiError, checkUpdate, NetworkError } from "../api/client.ts";
import type { UpdateError, UpdateStatus, UpdateSummary } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { hasUnsaved, setUnsaved } from "./unsaved.ts";
import { autoCheckEnded, checkDisabled, checkError, checkSuccess, CHECK_CURRENT_MS, CHECK_TIMEOUT_MS, checkText, checkTitle, checkView, clock, downloaded, entryLook, errorKey, failedStep, failureText, fraction, HOLDING, notesFor, pollDelay, runningVersion, shownError, statusRow, stepOf } from "./update.ts";

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
  checked_at: null,
  retry_at: null,
  checking: false,
  ...over,
});
const summary = (over: Partial<UpdateSummary> = {}): UpdateSummary => ({ current: "2.1.0", latest: "2.2.0", newer: true, can_apply: true, method: "setup", state: "idle", checked_at: null, retry_at: null, checking: false, ...over });
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
  assert.equal(pollDelay("idle", false, true), 1000);
  assert.equal(pollDelay("idle", false, false), 30_000);
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
  assert.ok(!/UpdatePanel|UpdateCard|upd-card|updSteps/.test(read("../components/settings-page.tsx")));
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

test("manual check tooltips use local days and the exact translated copy", () => {
  const now = new Date(2026, 9, 27, 10, 36).getTime();
  const at = (year: number, month: number, day: number, hour: number, minute: number) => new Date(year, month, day, hour, minute).toISOString();
  assert.equal(checkTitle(zh, null, now), "检查更新");
  assert.equal(checkTitle(zh, new Date(now - 30_000).toISOString(), now), "检查更新（上次：刚刚）");
  assert.equal(checkTitle(zh, at(2026, 9, 27, 10, 35), now), "检查更新（上次：今天 10:35）");
  assert.equal(checkTitle(zh, at(2026, 9, 27, 9, 36), now), "检查更新（上次：今天 09:36）");
  assert.equal(checkTitle(zh, at(2026, 9, 26, 9, 36), now), "检查更新（上次：昨天 09:36）");
  assert.equal(checkTitle(zh, at(2026, 9, 25, 9, 36), now), "检查更新（上次：10月25日 09:36）");
  assert.equal(checkTitle(zh, at(2025, 9, 25, 9, 36), now), "检查更新（上次：2025年10月25日 09:36）");
  assert.equal(checkTitle(en, at(2026, 9, 27, 9, 36), now), "Check for updates (last: today 09:36)");
  assert.equal(checkTitle(en, at(2026, 9, 25, 9, 36), now), "Check for updates (last: Oct 25 09:36)");
  assert.equal(checkTitle(ja, at(2026, 9, 26, 9, 36), now), "更新を確認（前回：昨日 09:36）");
  assert.equal(checkTitle(en, null, now), "Check for updates");
  assert.equal(checkTitle(ja, null, now), "更新を確認");
  assert.equal(zh.updChecking, "正在检查…");
  assert.equal(zh.updLatest("2.3.0"), "已是最新版本 2.3.0");
});

test("manual check rest, checking, four-second current and newer states stay distinct", () => {
  const now = Date.now();
  const rest = checkView({ kind: "rest" }, false, null, now);
  assert.deepEqual(rest, { phase: "rest" });
  assert.equal(checkDisabled(rest), false);
  assert.equal(checkText(zh, rest), null);
  const checking = checkView({ kind: "rest" }, true, null, now);
  assert.deepEqual(checking, { phase: "checking" });
  assert.equal(checkDisabled(checking), true);
  assert.equal(checkText(zh, checking), "正在检查…");
  const current = checkSuccess({ fresh: true, checked_at: null, update: status({ current: "2.3.0", latest: "2.3.0", newer: false }) }, now);
  assert.equal(current.open, false);
  assert.deepEqual(current.outcome, { kind: "current", version: "2.3.0", until: now + CHECK_CURRENT_MS });
  const shown = checkView(current.outcome, false, null, now + CHECK_CURRENT_MS - 1);
  assert.equal(checkText(zh, shown), "已是最新版本 2.3.0");
  assert.equal(checkDisabled(shown), false);
  assert.deepEqual(checkView(current.outcome, false, null, now + CHECK_CURRENT_MS), { phase: "rest" });
  const newerStatus = status({ current: "2.3.0", latest: "2.4.0", newer: true, can_apply: false });
  const newer = checkSuccess({ fresh: true, checked_at: null, update: newerStatus }, now);
  assert.equal(newer.open, true);
  assert.deepEqual(newer.outcome, { kind: "rest" });
  assert.equal(entryLook(summary({ can_apply: false }), newerStatus, null), "available");
});

test("an automatic check the page watched ends with the click's four-second current, news with none", () => {
  const now = Date.now();
  const ended = autoCheckEnded(status({ current: "2.3.0", latest: "2.3.0", newer: false }), now);
  assert.deepEqual(ended, { kind: "current", version: "2.3.0", until: now + CHECK_CURRENT_MS });
  assert.equal(checkText(zh, checkView(ended!, false, null, now + CHECK_CURRENT_MS - 1)), "已是最新版本 2.3.0");
  assert.deepEqual(checkView(ended!, false, null, now + CHECK_CURRENT_MS), { phase: "rest" });
  // News shows on the entry instead; without a status there is nothing to say.
  assert.equal(autoCheckEnded(status({ newer: true }), now), null);
  assert.equal(autoCheckEnded(null, now), null);
  // The hook watches the server's checking (never its own click's), and only a
  // check it showed the spinner for ends in the toast.
  const hook = read("../hooks/use-update.ts");
  assert.match(hook, /const autoChecking = useRef\(false\);/);
  assert.match(hook, /if \(localChecking\) return;\s*if \(serverChecking\) \{\s*autoChecking\.current = true;/);
  assert.match(hook, /if \(!autoChecking\.current\) return;\s*autoChecking\.current = false;\s*const ended = autoCheckEnded\(statusRef\.current, Date\.now\(\)\);/);
});

test("manual check offline and rate-limit errors expose host and exact re-enable times", () => {
  const now = new Date(2026, 9, 27, 10, 0).getTime();
  assert.deepEqual(checkError(new ApiError(502, { error: { code: "offline", reason: "timeout", host: "mirror.example" } })), { kind: "offline", host: "mirror.example" });
  assert.deepEqual(checkError(new ApiError(502, { error: { code: "offline", reason: "refused" } })), { kind: "offline", host: "github.com" });
  assert.deepEqual(checkError(new NetworkError()), { kind: "offline", host: "github.com" });
  assert.deepEqual(checkError(new DOMException("aborted", "AbortError")), { kind: "offline", host: "github.com" });
  const offline = checkView({ kind: "offline", host: "github.com" }, false, null, now);
  assert.equal(checkText(zh, offline), "无法连接 github.com");
  assert.equal(checkDisabled(offline), false);
  const retryAt = new Date(2026, 9, 27, 10, 36).toISOString();
  const retryMs = Date.parse(retryAt);
  assert.deepEqual(checkError(new ApiError(429, { error: { code: "rate_limited", retry_at: retryAt } })), { kind: "limited", retryAt });
  const limited = checkView({ kind: "limited", retryAt }, false, null, now);
  assert.deepEqual(limited, { phase: "limited", time: "10:36", tomorrow: false });
  assert.equal(checkText(zh, limited), "10:36 后再试");
  assert.equal(checkDisabled(limited), true);
  assert.deepEqual(checkView({ kind: "rest" }, false, retryAt, now), limited);
  assert.equal(checkDisabled(checkView({ kind: "limited", retryAt }, false, null, retryMs - 1)), true);
  const enabled = checkView({ kind: "limited", retryAt }, false, null, retryMs);
  assert.deepEqual(enabled, { phase: "rest" });
  assert.equal(checkDisabled(enabled), false);
  const tomorrow = checkView({ kind: "limited", retryAt: new Date(2026, 9, 28, 9, 0).toISOString() }, false, null, new Date(2026, 9, 27, 23, 0).getTime());
  assert.equal(checkText(zh, tomorrow), "明天 09:00 后再试");
  assert.deepEqual(checkError(new ApiError(409, { error: { code: "updating" } })), { kind: "updating" });
  assert.deepEqual(checkError(new ApiError(500, null)), { kind: "unknown" });
});

test("manual check API posts no body, passes its signal and keeps nested error bodies", async () => {
  const origin = "http://127.0.0.1:3600";
  const originalFetch = globalThis.fetch;
  const originalLocation = (globalThis as { location?: unknown }).location;
  const checkedAt = new Date(2026, 9, 27, 10, 36).toISOString();
  const seen: { url: string; init: RequestInit }[] = [];
  const controller = new AbortController();
  (globalThis as { location?: unknown }).location = { origin, href: `${origin}/` };
  try {
    globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
      seen.push({ url: String(url), init: init ?? {} });
      return new Response(JSON.stringify({ fresh: true, checked_at: checkedAt, update: status({ checked_at: checkedAt }) }), { status: 200, headers: { "content-type": "application/json" } });
    }) as typeof fetch;
    const answer = await checkUpdate(controller.signal);
    assert.equal(answer.fresh, true);
    assert.equal(answer.checked_at, checkedAt);
    assert.equal(answer.update.checked_at, checkedAt);
    const request = seen[0]!;
    assert.equal(request.url, `${origin}/api/update/check`);
    assert.equal(request.init.method, "POST");
    assert.equal(request.init.body, undefined);
    assert.equal(request.init.signal, controller.signal);
    assert.equal(new Headers(request.init.headers).get("content-type"), null);
    const refusals = [
      [429, { error: { code: "rate_limited", retry_at: checkedAt } }],
      [502, { error: { code: "offline", reason: "timeout", host: "github.com" } }],
      [409, { error: { code: "updating" } }],
    ] as const;
    for (const [httpStatus, body] of refusals) {
      globalThis.fetch = (async () => new Response(JSON.stringify(body), { status: httpStatus, headers: { "content-type": "application/json" } })) as typeof fetch;
      await assert.rejects(checkUpdate(), (error: unknown) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.status, httpStatus);
        assert.deepEqual(error.body, body);
        return true;
      });
    }
  } finally {
    globalThis.fetch = originalFetch;
    (globalThis as { location?: unknown }).location = originalLocation;
  }
});

test("manual check placement, sizing, request guards and single detail UI stay intact", () => {
  const component = read("../components/update.tsx");
  const manual = component.slice(component.indexOf("export function UpdateCheck"), component.indexOf("function Row("));
  assert.match(manual, /if \(manual === null\) return null/);
  assert.match(manual, /Icon name="ArrowClockwise"/);
  assert.ok(!/ArrowCounterClockwise|aria-haspopup|UpdatePanel|UpdateCard|upd-card|setOpen/.test(manual));
  assert.match(manual, /if \(long\.current\)[\s\S]*event\.preventDefault\(\);[\s\S]*return;/);
  for (const name of ["UpdateCard", "UpdatePanel", "UpdatePopover", "UpdateSheet"]) assert.equal(component.match(new RegExp(`function ${name}\\(`, "g"))?.length, 1);
  const icons = read("../components/icon-paths.ts");
  assert.match(icons, /ArrowClockwise: \["M240,56v48a8,8,0,0,1-8,8H184/);
  const shell = read("../components/shell.tsx");
  const buildStart = shell.indexOf('<p class="build-line"');
  const build = shell.slice(buildStart, shell.indexOf("</p>", buildStart));
  assert.ok(build.indexOf("Build:") < build.indexOf("{updateCheck}"));
  const settings = read("../components/settings-page.tsx");
  const indexStart = settings.indexOf('<p class="build-line is-index"');
  const index = settings.slice(indexStart, settings.indexOf("</p>", indexStart));
  assert.ok(index.indexOf("version.parts.version") < index.indexOf("{updateCheck}"));
  assert.ok(index.indexOf("version.parts.date") < index.indexOf("{updateCheck}"));
  assert.equal(settings.match(/<ul class="settings-index">/g)?.length, 1);
  const app = read("../app.tsx");
  assert.match(app, /updateCheck=\{caps\?\.update \? <UpdateCheck t=\{t\} update=\{update\} place="sidebar" \/> : undefined\}/);
  assert.match(app, /updateCheck=\{caps\?\.update \? <UpdateCheck t=\{t\} update=\{update\} place="phone" \/> : undefined\}/);
  assert.equal(app.match(/<UpdatePanel/g)?.length, 1);
  const css = read("../styles/app.css");
  const line = css.slice(css.indexOf(".build-line {"), css.indexOf("}", css.indexOf(".build-line {")));
  assert.match(line, /display: flex;/);
  assert.match(line, /align-items: center;/);
  assert.match(line, /justify-content: center;/);
  assert.match(line, /white-space: nowrap;/);
  // a 28 px target everywhere, without making the 11 px line taller; the status hides the line's text
  assert.match(css, /\.update-check \{[^}]*width: 28px;\s*height: 28px;\s*margin: -6px 0;/);
  assert.match(css, /\.build-line:has\(\.update-check-status\) \.build-copy \{\s*display: none;/);
  const hook = read("../hooks/use-update.ts");
  assert.match(hook, /if \(!enabled \|\| disabled \|\| checkBusy\.current \|\| serverCheckingRef\.current\) return;\s*checkBusy\.current = true;/);
  assert.match(hook, /checkEpoch\.current \+= 1/);
  // Disabled while an update runs, never hidden by the server's `update.check` (only by no `Capabilities.update`).
  assert.match(hook, /const disabled = checkDisabled\(view\) \|\| updating;/);
  assert.ok(!/manual: [^\n]*(newer|automatic|check_enabled)/.test(hook));
  assert.match(hook, /if \(checkBusy\.current \|\| checkEpoch\.current !== epoch\) return null/);
  assert.match(hook, /reconnecting \|\| localChecking\) return/);
  assert.match(hook, /next\?\.checking \?\? serverChecking/);
  assert.match(hook, /setTimeout\(\(\) => controller\.abort\(\), CHECK_TIMEOUT_MS\)/);
  assert.equal(CHECK_TIMEOUT_MS, 12_000);
  assert.match(hook, /setStatus\(answer\.update\)/);
  assert.match(hook, /if \(applied\.open\) \{[\s\S]*?setOpen\(true\)/);
  assert.match(hook, /failure\.kind === "updating"[\s\S]*?refreshAfter = true/);
  assert.match(hook, /manual: enabled \? \{ view, disabled, checkedAt, now, run \} : null/);
});
