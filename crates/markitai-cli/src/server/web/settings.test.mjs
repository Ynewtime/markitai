import test from 'node:test';
import assert from 'node:assert/strict';
import {credentialFields} from './settings.js';
import {useLocale} from './i18n.js';

const inputs = (key, base = ['keep', '']) => ({key: {mode: key[0], value: key[1]}, base: {mode: base[0], value: base[1]}});

test('Keep sends nothing, Clear sends null and Replace sends the trimmed value', () => {
  useLocale('en');
  assert.deepEqual(credentialFields(inputs(['keep', 'ignored']), true), {});
  assert.deepEqual(credentialFields(inputs(['clear', ''], ['clear', '']), true), {api_key: null, api_base: null});
  assert.deepEqual(credentialFields(inputs(['replace', '  sk-test  '], ['replace', ' http://127.0.0.1:9/v1 ']), true), {api_key: 'sk-test', api_base: 'http://127.0.0.1:9/v1'});
});

test('editing a saved connection still demands the value a Replace choice promises', () => {
  useLocale('en');
  assert.throws(() => credentialFields(inputs(['replace', '   ']), true), /Enter an API key/);
  assert.throws(() => credentialFields(inputs(['keep', ''], ['replace', '']), true), /Enter a base URL/);
});

test('a new connection starts in Replace, and an empty key then relies on the environment instead of failing', () => {
  useLocale('en');
  // "+ Add connection" leaves the key field enabled in Replace; submitting without a key is allowed.
  assert.deepEqual(credentialFields(inputs(['replace', '']), false), {});
  assert.deepEqual(credentialFields(inputs(['replace', 'sk-test']), false), {api_key: 'sk-test'});
  // A base URL chosen for replacement is still required.
  assert.throws(() => credentialFields(inputs(['replace', ''], ['replace', '']), false), /Enter a base URL/);
});

// A document that records what the editor does to its controls; ids not named here are stand-ins.
function stubDocument() {
  const nodes = new Map();
  const make = id => ({id, value: '', disabled: false, hidden: false, textContent: '', title: '', dataset: {}, listeners: {},
    addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
    click() { for (const handler of this.listeners.click ?? []) handler({preventDefault() {}}); },
    removeAttribute() {}, setAttribute() {}, replaceChildren() {}, append() {}, focus() {}, scrollIntoView() {},
    getBoundingClientRect: () => ({top: 0}), classList: {toggle() {}}, querySelectorAll: () => []});
  return {getElementById: id => { if (!nodes.has(id)) nodes.set(id, make(id)); return nodes.get(id); }};
}

test('"+ Add connection" and "Reset draft" leave the API key input enabled, in Replace', async () => {
  globalThis.document = stubDocument();
  globalThis.innerHeight = 800;
  const {initSettings} = await import('./settings.js');
  initSettings({notice() {}, confirmDelete: async () => true, onSaved: async () => {}, routable: () => false});
  const $ = id => document.getElementById(id);
  // The state a page is in before: Keep, input disabled (as clearSecrets leaves it after an edit or a save).
  const locked = () => { $('key-mode').value = 'keep'; $('provider-key').disabled = true; };
  for (const control of ['connection-new', 'connection-reset']) {
    locked();
    $(control).click();
    assert.equal($('key-mode').value, 'replace', control);
    assert.equal($('provider-key').disabled, false, control);
    assert.equal($('provider-key').value, '', control);
    // The base URL keeps its default: nothing to replace unless asked.
    assert.equal($('base-mode').value, 'keep', control);
    assert.equal($('provider-base').disabled, true, control);
  }
  // Choosing Keep again still locks the field, as before.
  $('key-mode').value = 'keep'; for (const handler of $('key-mode').listeners.change) handler();
  assert.equal($('provider-key').disabled, true);
  delete globalThis.document; delete globalThis.innerHeight;
});
