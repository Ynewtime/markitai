import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {DICTIONARIES, detectLocale, t, useLocale, currentLocale, nextTheme} from './i18n.js';

const placeholders = text => [...text.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort();

test('both languages define the same keys with the same placeholders', () => {
  const {en, zh} = DICTIONARIES;
  assert.deepEqual(Object.keys(zh).sort(), Object.keys(en).sort());
  for (const key of Object.keys(en)) {
    assert.equal(typeof zh[key], 'string', key);
    assert.ok(zh[key].length > 0, key);
    assert.deepEqual(placeholders(zh[key]), placeholders(en[key]), key);
  }
});

test('every key the page and scripts use exists in the dictionaries', () => {
  const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');
  const html = read('./index.html');
  const used = new Set([...html.matchAll(/data-i18n(?:-[a-z-]+)?="([^"]+)"/g)].map(match => match[1]));
  for (const name of ['app.js', 'settings.js', 'result-tools.js', 'preview.js', 'api.js']) {
    for (const match of read(`./${name}`).matchAll(/\b(?:t|label)\((?:[^,()]+,\s*)?'([a-zA-Z]+)'/g)) used.add(match[1]);
  }
  assert.ok(used.size > 100, `only ${used.size} keys found`);
  for (const key of used) assert.ok(key in DICTIONARIES.en || `${key}_one` in DICTIONARIES.en, `missing key ${key}`);
});

test('an explicit choice wins, otherwise the browser language decides', () => {
  assert.equal(detectLocale('zh', 'en-US'), 'zh');
  assert.equal(detectLocale('en', 'zh-CN'), 'en');
  assert.equal(detectLocale(null, 'zh-TW'), 'zh');
  assert.equal(detectLocale('fr', 'de-DE'), 'en');
  assert.equal(detectLocale(undefined, undefined), 'en');
});

test('interpolation, plural forms and fallbacks', () => {
  try {
    useLocale('en');
    assert.equal(t('jobAttention', {count: 1}), '1 item needs attention.');
    assert.equal(t('jobAttention', {count: 3}), '3 items need attention.');
    assert.equal(t('progressCount', {done: 2, total: 5}), '2 of 5 finished');
    assert.equal(t('fetchFailed', {}), 'Could not fetch this page: {error}');
    assert.equal(t('no-such-key'), 'no-such-key');
    useLocale('zh');
    assert.equal(currentLocale(), 'zh');
    assert.equal(t('jobAttention', {count: 1}), '有 1 项需要处理。');
    assert.equal(t('diffSummary', {added: 2, removed: 1}), '基础版 → 增强版 · 新增 2 行，删除 1 行');
    useLocale('xx');
    assert.equal(currentLocale(), 'en');
  } finally { useLocale('en'); }
});

test('the theme control cycles through automatic, light and dark', () => {
  assert.equal(nextTheme('auto'), 'light');
  assert.equal(nextTheme('light'), 'dark');
  assert.equal(nextTheme('dark'), 'auto');
  assert.equal(nextTheme('unknown'), 'auto');
});
