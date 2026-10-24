import assert from "node:assert/strict";
import test from "node:test";
import { BRAND_MARKS } from "../components/brand-marks.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { refusedHosts } from "./models.ts";

test("a refused key names the hosts the server tried, else the one the page asked", () => {
  assert.deepEqual(refusedHosts(["https://api.moonshot.ai/v1", "https://api.moonshot.cn/v1"], "https://api.moonshot.ai/v1"), [
    "api.moonshot.ai",
    "api.moonshot.cn",
  ]);
  // An older backend names none: the page's own address.
  assert.deepEqual(refusedHosts(undefined, "https://api.deepseek.com"), ["api.deepseek.com"]);
  assert.deepEqual(refusedHosts([], "http://127.0.0.1:8000/v1"), ["127.0.0.1:8000"]);
  assert.deepEqual(refusedHosts(undefined, ""), []);
});

test("Moonshot's and Kimi Code's refusals say which endpoint refused and where the key belongs", () => {
  const both = ["api.moonshot.ai", "api.moonshot.cn"];
  assert.equal(
    en.discoveryFailHint("key_rejected", null, both, "moonshot"),
    "api.moonshot.ai and api.moonshot.cn refused the key. Check that it is correct and still active. A Kimi Code membership key belongs under Kimi Code.",
  );
  assert.equal(
    zh.discoveryFailHint("key_rejected", null, both, "moonshot"),
    "api.moonshot.ai 和 api.moonshot.cn 拒绝了这个密钥。请确认密钥正确且仍然有效。Kimi Code 会员的密钥请在「Kimi Code」中添加。",
  );
  assert.ok(ja.discoveryFailHint("key_rejected", null, both, "moonshot").startsWith("api.moonshot.ai と api.moonshot.cn がキーを拒否しました。"));
  for (const dict of [en, zh, ja]) {
    assert.ok(dict.discoveryFailHint("key_rejected", null, ["api.kimi.com"], "kimi_code").includes("Moonshot AI"));
    // Other providers keep the plain advice, now naming the host.
    const deepseek = dict.discoveryFailHint("key_rejected", null, ["api.deepseek.com"], "deepseek");
    assert.ok(deepseek.startsWith("api.deepseek.com") && !deepseek.includes("Kimi"), deepseek);
    // A caller that names no host (an older page) reads as before.
    assert.equal(dict.discoveryFailHint("key_rejected", null), dict.discoveryFailHint("key_rejected", null, [], ""));
    // A model test names both regions' addresses for a Moonshot model.
    const tested = dict.testFailHint("key_rejected", null, "moonshot/kimi-k3");
    assert.ok(tested.includes("https://api.moonshot.cn/v1") && tested.includes("https://api.moonshot.ai/v1"), tested);
    assert.ok(!dict.testFailHint("key_rejected", null, "openai/gpt-x").includes("moonshot"));
    assert.ok(dict.regionMoved("https://api.moonshot.cn/v1").includes("https://api.moonshot.cn/v1"));
    assert.ok(dict.regionElsewhere("https://api.moonshot.cn/v1").includes("https://api.moonshot.cn/v1"));
  }
});

test("Kimi Code has its own brand mark beside Moonshot's", () => {
  assert.ok(BRAND_MARKS.kimi && BRAND_MARKS.moonshot);
  assert.ok(BRAND_MARKS.kimi.paths[0] !== BRAND_MARKS.moonshot.paths[0]);
  assert.equal(BRAND_MARKS.kimi.viewBox, "0 0 24 24");
});
