import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, readdirSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {DICTIONARIES, detectLocale, t, useLocale, currentLocale, nextTheme, messageKeys, apiErrorMessage, itemErrorMessage, serviceNote, persistenceMessage, discoveryStatus, providerName} from './i18n.js';

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
    assert.equal(t('unsupportedFormat', {}), 'This file type is not supported: {format}.');
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

const source = name => readFileSync(new URL(name, import.meta.url), 'utf8');
const inZh = action => { try { useLocale('zh'); return action(); } finally { useLocale('en'); } };

test('every service message the tables can produce exists in both languages', () => {
  const keys = messageKeys();
  assert.ok(keys.length > 70, `only ${keys.length} keys`);
  for (const key of keys) {
    assert.ok(key in DICTIONARIES.en, `missing en ${key}`);
    assert.ok(key in DICTIONARIES.zh, `missing zh ${key}`);
  }
});

test('every reason the Rust service sends and every core error code has localized text', () => {
  const rust = ['http.rs', 'types.rs', 'security.rs', 'files.rs', 'store.rs', 'rerun.rs', 'jobs.rs', 'providers.rs', 'settings.rs',
    ...readdirSync(new URL('../settings/', import.meta.url)).map(name => `settings/${name}`)].map(name => source(`../${name}`)).join('\n');
  const reasons = new Set([...rust.matchAll(/(?:ApiError|Self)::(?:new|structured)\(\s*\d+,\s*"([a-z_]+)"/g)].map(match => match[1]));
  for (const reason of ['request_too_large', 'invalid_multipart']) { assert.ok(rust.includes(`"${reason}"`)); reasons.add(reason); }
  assert.ok(reasons.size > 50, `only ${reasons.size} reasons found`);
  // Routing and startup errors keep the generic status text by design.
  const generic = new Set(['route_not_found', 'method_not_allowed', 'invalid_allowed_host']);
  for (const reason of reasons) {
    if (generic.has(reason)) continue;
    assert.notEqual(apiErrorMessage(400, {detail: 'service words', reason}).text, 'service words', `unmapped reason ${reason}`);
  }
  const core = source('../../../../markitai-core/src/types.rs');
  const codes = new Set([...core.slice(core.indexOf('pub fn code'), core.indexOf('pub type Result')).matchAll(/=> "([a-z_]+)"/g)].map(match => match[1]));
  assert.ok(codes.size >= 9, [...codes].join());
  // Service causes an item can carry, including errors raised inside a retry worker.
  for (const code of ['cancelled', 'shutdown', 'internal_error', 'enhancement_failed', 'no_output', 'output_conflict', 'file_not_found', 'output_identity_conflict']) {
    assert.ok(rust.includes(`"${code}"`), code); codes.add(code);
  }
  for (const code of codes) assert.notEqual(itemErrorMessage({error: 'opaque words', error_code: code}).text, 'opaque words', `unmapped item code ${code}`);
});

test('the English phrases the page recognizes are still the ones the core and service write', () => {
  const tree = relative => readdirSync(new URL(relative, import.meta.url), {recursive: true}).filter(name => name.endsWith('.rs')).map(name => source(`${relative}${name}`)).join('\n');
  const rust = tree('../../../../markitai-core/src/') + tree('../');
  for (const phrase of [
    'No model configured', 'Unsupported file format: ', ' Supported extensions: ', 'Local OCR requires macOS', 'Local OCR backend is unavailable',
    'LLM returned HTTP {status}', 'the model is not available in this region', "the account's quota or billing does not allow this request", 'the model is unavailable',
    'LLM request timed out', 'LLM request failed', 'HTTP {}', 'Remote fetching is disabled by policy', 'URL returned no extractable content',
    'Model connection test timed out', 'Model connection request failed', 'Model connection returned HTTP {status}', 'An API endpoint is required', '{} responded',
    'Model credentials or endpoint configuration are invalid or unavailable', 'This model provider is not supported by the native runtime', 'Model connection test failed',
    'Model discovery failed; check endpoint', 'Refresh failed; showing previously discovered models', 'Model discovery wait timed out', 'Too many model discovery requests are active',
    'Models reported by the authenticated official Copilot runtime', 'Official Copilot runtime or authentication is unavailable',
    'cancelled (stopped by request)', 'cancelled (server shutdown)', 'LLM enhancement did not produce an enhanced result',
    'history could not be persisted; completed artifacts remain available until shutdown', 'output rollback failed; restart to recover the previous result', 'item deletion rollback failed; restart to recover',
  ]) assert.ok(rust.includes(phrase), `no Rust source writes "${phrase}" any more`);
});

test('item errors are localized by recognizable shape or code, with the original kept as detail', () => {
  const cases = [
    [{error: 'cancelled (stopped by request)'}, 'Stopped before conversion. Retry to convert it.', '', true],
    [{error: 'cancelled (server shutdown)', error_code: 'shutdown'}, 'Cancelled because the service stopped. Retry to convert it.', '', true],
    [{error: 'No model configured; set MODEL and a provider API key, or llm.model_list', error_code: 'no_model_configured'}, 'No model is configured. Add one under Connections, or set MODEL and a provider API key.'],
    [{error: 'Local OCR requires macOS 11 or later; no local OCR backend is available on this platform'}, 'Local OCR is not available on this system.'],
    [{error: 'HTTP 404', kind: 'url', error_code: 'fetch_error'}, 'The page was not found (HTTP 404).'],
    [{error: 'HTTP 403', kind: 'url'}, 'The website refused access (HTTP 403).'],
    [{error: 'HTTP 503', kind: 'url'}, 'The website had a server error (HTTP 503).'],
    [{error: 'HTTP 429', kind: 'url'}, 'The website is limiting requests (HTTP 429). Try again later.'],
    [{error: 'HTTP 302', kind: 'url'}, 'The website answered HTTP 302.'],
    [{error: 'error sending request: client error (Connect): tcp connect error: Connection refused (os error 61)', kind: 'url', error_code: 'fetch_error'}, "Could not connect to this website. Check the address and the server's network access."],
    [{error: 'operation timed out', kind: 'url'}, 'Fetching this page took too long.'],
    [{error: 'Office export timed out', kind: 'file'}, 'The conversion took too long and was stopped.'],
    [{error: 'LLM returned HTTP 401', error_code: 'conversion_error'}, 'The provider rejected the credentials (HTTP 401). Check the API key.'],
    [{error: 'LLM returned HTTP 403: the model is not available in this region'}, 'The provider does not offer this model in your region (HTTP 403). Choose another provider or model.'],
    [{error: "LLM returned HTTP 429: the account's quota or billing does not allow this request"}, 'The provider account has no remaining quota or needs billing set up (HTTP 429).'],
    [{error: 'LLM returned HTTP 404: the model is unavailable'}, 'The provider does not offer this model (HTTP 404). Check the model identifier.'],
    [{error: 'LLM returned HTTP 403: something new'}, 'The provider rejected the credentials (HTTP 403). Check the API key.'],
    [{error: 'LLM request timed out'}, 'The model did not respond in time.'],
    [{error: 'LLM returned no text'}, 'The model could not complete this document.'],
    [{error: 'Input exceeds the 500 MiB limit', error_code: 'invalid_input'}, 'The input is larger than a supported limit (500 MiB).'],
    [{error: 'Remote fetching is disabled by policy', kind: 'url', error_code: 'fetch_error'}, 'Remote fetching is turned off in the service configuration.'],
    [{error: 'URL returned no extractable content', kind: 'url'}, 'The page had no content that could be extracted.'],
    [{error: 'Malformed XML: unexpected end', error_code: 'conversion_error'}, 'The document could not be converted.'],
    [{error: 'retry would replace another item\'s artifact', error_code: 'output_conflict'}, "This item's output files overlap another item's, so the service left them unchanged."],
    [{error: 'internal conversion error', error_code: 'internal_error'}, 'The service hit an internal error. Try again; if it keeps happening, check the server log.'],
  ];
  for (const [item, text, detail = item.error, hint] of cases) {
    const result = itemErrorMessage(item);
    assert.equal(result.text, text, item.error);
    assert.equal(result.detail, detail, item.error);
    assert.equal(!!result.hint, !!hint, item.error);
  }
  // A message the page cannot classify, from an older history without a code, is shown unchanged.
  assert.deepEqual(itemErrorMessage({error: 'something entirely new'}), {text: 'something entirely new', detail: '', formats: ''});
  const unsupported = itemErrorMessage({error: "Unsupported file format: '.xyz'. Supported extensions: .csv .docx .pdf.", error_code: 'unsupported'});
  assert.deepEqual(unsupported, {text: "This file type is not supported: '.xyz'.", detail: '', formats: '.csv .docx .pdf'});
  inZh(() => {
    assert.equal(itemErrorMessage({error: 'HTTP 404', kind: 'url'}).text, '网页不存在（HTTP 404）。');
    assert.equal(itemErrorMessage({error: 'LLM returned HTTP 403: the model is not available in this region'}).text, '服务商在你所在的地区不提供这个模型（HTTP 403）。请换用其它服务商或模型。');
    assert.equal(itemErrorMessage({error: 'cancelled (stopped by request)', error_code: 'cancelled'}).text, '在开始转换前已停止。可以重试。');
    assert.equal(itemErrorMessage({error: "Unsupported file format: '.xyz'. Supported extensions: .csv.", error_code: 'unsupported'}).text, "不支持这种文件类型：'.xyz'。");
    assert.equal(itemErrorMessage({error: 'Native PDF conversion failed: broken xref', error_code: 'conversion_error'}).text, '无法转换这个文档。');
  });
});

test('provider probe and discovery phrases are translated; proper nouns and unknown text are kept', () => {
  assert.deepEqual(serviceNote('openai/gpt-x responded'), {text: 'openai/gpt-x responded.', detail: 'openai/gpt-x responded'});
  assert.equal(serviceNote('Model connection returned HTTP 404').text, 'The model or its endpoint was not found (HTTP 404). Check the model identifier and base URL.');
  assert.deepEqual(serviceNote('Model connection returned HTTP 403: the model is not available in this region'), {text: 'The provider does not offer this model in your region (HTTP 403). Choose another provider or model.', detail: 'Model connection returned HTTP 403: the model is not available in this region'});
  assert.equal(serviceNote('Model connection test timed out').text, 'The model did not respond in time.');
  assert.equal(serviceNote('Model connection response is not valid JSON').text, 'The model endpoint returned a response that could not be read.');
  assert.equal(serviceNote('Model discovery failed; check endpoint, credentials and provider availability').text, 'Model discovery failed. Check the endpoint, credentials and provider availability.');
  assert.deepEqual(serviceNote('A brand new provider phrase'), {text: 'A brand new provider phrase', detail: ''});
  assert.deepEqual(serviceNote(undefined), {text: '', detail: ''});
  inZh(() => {
    assert.equal(serviceNote('anthropic/claude-x responded').text, 'anthropic/claude-x 已响应。');
    assert.equal(serviceNote('Model connection returned HTTP 401').text, '服务商拒绝了凭据（HTTP 401）。请检查 API key。');
    assert.equal(serviceNote('Official Copilot runtime or authentication is unavailable').text, '官方 Copilot 运行时或其登录不可用。');
    assert.equal(serviceNote('Models reported by the authenticated official Claude runtime').text, '由已登录的官方 Claude 运行时报告的模型。');
    assert.equal(discoveryStatus('partial'), '部分发现');
    assert.equal(discoveryStatus('later'), 'later');
    assert.equal(providerName({provider: 'custom', label: 'OpenAI-compatible endpoint'}), 'OpenAI 兼容');
    assert.equal(providerName({provider: 'zz', label: 'Unknown provider'}), '未知服务商 (zz)');
    assert.equal(providerName({provider: 'deepseek', label: 'DeepSeek'}), 'DeepSeek');
    assert.equal(persistenceMessage('history could not be persisted; completed artifacts remain available until shutdown').text, '这个任务无法保存到历史。已完成的文件在服务停止前仍可下载。');
  });
  assert.deepEqual(persistenceMessage('a new persistence failure'), {text: 'a new persistence failure', detail: ''});
});

// boot.js runs before first paint; evaluate it against a minimal document.
function boot({stored = null, language = 'en-US', storageThrows = false} = {}) {
  const attributes = new Map(), timers = [];
  const root = {lang: '', setAttribute: (name, value) => attributes.set(name, value), removeAttribute: name => attributes.delete(name)};
  const document = {documentElement: root, title: 'Markitai · Documents, made useful'};
  const localStorage = {getItem: key => { if (storageThrows) throw new Error('blocked'); return key === 'markitai.lang' ? stored : null; }};
  runInNewContext(source('./boot.js'), {document, localStorage, navigator: {language}, setTimeout: (fn, delay) => timers.push([fn, delay]), String});
  return {root, attributes, document, timers};
}

test('a Chinese first paint hides the English static text until translation, with a timed fallback', () => {
  const zh = boot({language: 'zh-CN'});
  assert.equal(zh.root.lang, 'zh-CN');
  assert.ok(zh.attributes.has('data-i18n-pending'));
  assert.equal(zh.document.title, DICTIONARIES.zh.pageTitle);
  assert.equal(zh.timers.length, 1); assert.equal(zh.timers[0][1], 3000);
  zh.timers[0][0](); assert.ok(!zh.attributes.has('data-i18n-pending'));
  for (const options of [{language: 'en-US'}, {stored: 'en', language: 'zh-CN'}, {storageThrows: true, language: 'fr'}]) {
    const page = boot(options);
    assert.equal(page.root.lang, 'en'); assert.ok(!page.attributes.has('data-i18n-pending')); assert.equal(page.timers.length, 0);
  }
  assert.ok(boot({stored: 'zh', language: 'en-US'}).attributes.has('data-i18n-pending'));
  assert.ok(boot({storageThrows: true, language: 'zh-TW'}).attributes.has('data-i18n-pending'));
  // The page hides only while marked, i18n.js clears the mark, and every script stays external for the CSP.
  assert.match(source('./style.css'), /html\[data-i18n-pending\] \[data-i18n\][^{]*::placeholder\{color:transparent!important\}/);
  assert.match(source('./i18n.js'), /removeAttribute\('data-i18n-pending'\)/);
  const html = source('./index.html');
  for (const tag of html.match(/<script\b[^>]*>/g)) assert.match(tag, /\ssrc="\/ui\/[a-z-]+\.js"/);
  assert.match(html, /<progress id="upload-bar"[^>]*aria-labelledby="upload-progress-text"/);
});
