import assert from "node:assert/strict";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { withLead } from "./toast.ts";

test("a toast's lead and its text are joined in the language's own punctuation", () => {
  const model = "openai/fixture-model";
  assert.equal(withLead(en.modelTestPassed, en.modelTestReady(model), en.toastLead), "Model test passed. openai/fixture-model is ready to use.");
  assert.equal(withLead(zh.modelTestPassed, zh.modelTestReady(model), zh.toastLead), "模型测试通过。openai/fixture-model 已连接并可正常使用。");
  assert.equal(withLead(ja.modelTestPassed, ja.modelTestReady(model), ja.toastLead), "モデルのテストに成功しました。openai/fixture-model を使用できます。");
  assert.equal(withLead(zh.retryFailed, "原始上传已不在服务器上。", zh.toastLead), "重试失败。原始上传已不在服务器上。");
  // Without a text the lead stands alone.
  assert.equal(withLead(ja.stopFailed, "", ja.toastLead), ja.stopFailed);
  // Chinese and Japanese never get an ASCII full stop between the two.
  for (const dict of [zh, ja]) assert.match(dict.toastLead, /^[^\x00-\x7f]+$/);
  // English toasts that carry a lead and a detail in one string use sentences, not "·".
  for (const text of [en.restoreFailed, en.draftKept]) assert.match(text, /^[^·]+\. [A-Z][^·]+\.$/);
});

test("a page that could not be kept says why in words that hold for every page", () => {
  for (const dict of [en, zh, ja]) {
    const blank = dict.pageNotKept("empty", "blank");
    const notice = dict.pageNotKept("empty", "javascript");
    // A blank page is not said to be built by script: a short page may be one picture and a caption.
    assert.equal(/script|脚本|スクリプト/i.test(blank), false);
    assert.match(notice, /JavaScript/);
    assert.notEqual(blank, notice);
    // A backend that sends no reason reads as blank.
    assert.equal(dict.pageNotKept("empty", null), blank);
    assert.notEqual(dict.pageNotKept("challenge", "challenge"), blank);
  }
});
