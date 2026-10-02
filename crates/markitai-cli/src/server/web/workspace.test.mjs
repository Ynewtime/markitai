import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {BOOLEAN_OPTIONS, TEXT_OPTIONS, LLM_OPTIONS, OPTION_CAPTIONS, OPTION_HELP, OPTIONS_KEY, publicOptions, readOptions, writeOptions, presetNeedsModel, isHiddenName, filePath, selectFolderFiles, walkEntries, ledgerState, ledgerCounts, ledgerItems, isUnsupported, canRetry, failedItems, hasOutput, historyParts, routeView, viewPath} from './workspace.js';
import {mergeFiles} from './api.js';
import {DICTIONARIES, t, useLocale} from './i18n.js';

const memory = (initial) => { const map = new Map(initial ? [[OPTIONS_KEY, initial]] : []); return {map, getItem: key => map.get(key) ?? null, setItem: (key, value) => map.set(key, value), removeItem: key => map.delete(key)}; };

test('only public options with valid values survive, from a job or from storage', () => {
  assert.deepEqual(publicOptions({preset: 'standard', llm: true, ocr: false, profile: 'rag', strategy: 'static', backend: 'native', origin: 'cli', secret: 'x'}),
    {preset: 'standard', llm: true, ocr: false, profile: 'rag', strategy: 'static', backend: 'native'});
  assert.deepEqual(publicOptions({llm: 'true', alt: 1, profile: 'markdown', strategy: 'fast', backend: 'x', preset: ''}), {});
  assert.deepEqual(publicOptions({preset: 'my preset'}), {preset: 'my preset'});
  assert.deepEqual(publicOptions({preset: 'x'.repeat(65)}), {});
  for (const bad of [null, undefined, 'text', 7, [1]]) assert.deepEqual(publicOptions(bad), {});
  assert.equal(BOOLEAN_OPTIONS.length + TEXT_OPTIONS.length, 13);
  assert.deepEqual(LLM_OPTIONS, ['llm', 'alt', 'desc']);
});

test('the last options are remembered per browser and a blocked or damaged store means defaults', () => {
  const store = memory();
  writeOptions({llm: true, preset: 'rich', junk: 1}, store);
  assert.deepEqual(JSON.parse(store.map.get(OPTIONS_KEY)), {llm: true, preset: 'rich'});
  assert.deepEqual(readOptions(store), {llm: true, preset: 'rich'});
  // Choosing the defaults again removes the entry instead of keeping an empty object.
  writeOptions({}, store);
  assert.equal(store.map.has(OPTIONS_KEY), false);
  assert.deepEqual(readOptions(store), {});
  for (const damaged of ['{broken', '[1]', 'null', '"x"', '{"llm":"yes","profile":"nope"}']) assert.deepEqual(readOptions(memory(damaged)), {}, damaged);
  const blocked = {getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); }, removeItem() { throw new Error('blocked'); }};
  assert.deepEqual(readOptions(blocked), {});
  assert.doesNotThrow(() => writeOptions({llm: true}, blocked));
  assert.deepEqual(readOptions(null), {});
  assert.doesNotThrow(() => writeOptions({llm: true}, null));
});

test('a preset needs a model when its definition turns the LLM on', () => {
  const capabilities = {preset_options: {minimal: {llm: false}, standard: {llm: true}, rich: {llm: true}, offline: {llm: false, ocr: true}}};
  assert.deepEqual(['minimal', 'standard', 'rich', 'offline', 'unknown'].map(name => presetNeedsModel(name, capabilities)), [false, true, true, false, false]);
  assert.equal(presetNeedsModel('standard', null), false);
});

test('every option has a caption, an explanation and a tooltip text in both languages', () => {
  for (const key of [...BOOLEAN_OPTIONS, ...TEXT_OPTIONS]) {
    assert.ok(OPTION_HELP[key], `no help for ${key}`);
    for (const language of ['en', 'zh']) assert.ok(DICTIONARIES[language][OPTION_HELP[key]]?.length > 12,`${language} ${OPTION_HELP[key]}`);
  }
  for (const key of BOOLEAN_OPTIONS) for (const language of ['en', 'zh']) assert.ok(DICTIONARIES[language][OPTION_CAPTIONS[key]], `${language} ${OPTION_CAPTIONS[key]}`);
  for (const key of ['needsModel', 'presetNeedsModel', 'llmReady_one', 'llmReady_other', 'llmReadyPlain', 'llmMissing', 'llmConfigure', 'optionsReset', 'optionsChanged_one', 'optionsChanged_other']) {
    for (const language of ['en', 'zh']) assert.ok(DICTIONARIES[language][key], `${language} ${key}`);
  }
});

test('hidden and system files are skipped inside a chosen folder only', () => {
  for (const name of ['.DS_Store', '.git', '.hidden.md', 'Thumbs.db', 'thumbs.DB', 'desktop.ini']) assert.equal(isHiddenName(name), true, name);
  for (const name of ['report.pdf', 'notes.md', 'a.b.c', 'Thumbs.db.txt', 'my.dot']) assert.equal(isHiddenName(name), false, name);
  const file = (path, name = path.split('/').pop()) => ({name, size: 1, webkitRelativePath: path.includes('/') ? path : ''});
  const chosen = selectFolderFiles([file('docs/a.pdf'), file('docs/.DS_Store'), file('docs/sub/Thumbs.db'), file('docs/.git/config', 'config'), file('docs/sub/b.md'), file('.notes/x.md', 'x.md'), file('single.txt')]);
  assert.deepEqual(chosen.files.map(filePath), ['docs/a.pdf', 'docs/sub/b.md', '.notes/x.md', 'single.txt']);
  assert.equal(chosen.hidden, 3);
  // The folder the person chose is never filtered, even when its own name starts with a dot.
  assert.deepEqual(selectFolderFiles([file('.notes/x.md', 'x.md'), file('.notes/.secret', '.secret')]).files.map(filePath), ['.notes/x.md']);
});

// Minimal stand-ins for the FileSystemEntry API, including its batched directory reads.
const fakeFile = (name, size = 1) => ({isFile: true, isDirectory: false, name, file: done => done({name, size})});
const brokenFile = name => ({isFile: true, isDirectory: false, name, file: (_ok, fail) => fail(new Error('denied'))});
const fakeDir = (name, children, batch = 2) => ({isFile: false, isDirectory: true, name, createReader() { let at = 0; return {readEntries(done) { const part = children.slice(at, at + batch); at += batch; done(part); }}; }});

test('a dropped folder is walked in order, with hidden files skipped and relative paths kept', async () => {
  const root = fakeDir('docs', [fakeFile('b.pdf'), fakeFile('.DS_Store'), fakeDir('sub', [fakeFile('c.md'), fakeFile('Thumbs.db'), fakeDir('.git', [fakeFile('HEAD')])]), fakeFile('a.docx'), brokenFile('locked.pdf')]);
  const found = await walkEntries([root, fakeFile('.env')]);
  assert.deepEqual(found.files.map(filePath), ['docs/a.docx', 'docs/b.pdf', 'docs/sub/c.md', '.env']);
  assert.deepEqual([found.hidden, found.unreadable, found.truncated], [3, 1, false]);
  // The path is a property of the selection, not a rename: the File keeps its own name.
  assert.equal(found.files[2].name, 'c.md');
});

test('walking stops at the item limit and says so, and an enormous tree cannot hold the page', async () => {
  const many = fakeDir('big', Array.from({length: 30}, (_, index) => fakeFile(`f${String(index).padStart(2, '0')}.txt`)), 7);
  const limited = await walkEntries([many], {limit: 10});
  assert.equal(limited.files.length, 10);
  assert.equal(limited.truncated, true);
  assert.deepEqual(limited.files.slice(0, 3).map(filePath), ['big/f00.txt', 'big/f01.txt', 'big/f02.txt']);
  const exact = await walkEntries([fakeDir('d', [fakeFile('a'), fakeFile('b')])], {limit: 2});
  assert.deepEqual([exact.files.length, exact.truncated], [2, false]);
  const none = await walkEntries([fakeDir('d', [fakeFile('a')])], {limit: 0});
  assert.deepEqual([none.files.length, none.truncated], [0, true]);
  const visited = await walkEntries([many], {visits: 5});
  assert.equal(visited.truncated, true);
  let nested = fakeFile('deep.txt');
  for (let level = 0; level < 100; level++) nested = fakeDir(`d${level}`, [nested]);
  const deep = await walkEntries([nested]);
  assert.deepEqual([deep.files.length, deep.truncated], [0, true]);
});

test('files in different folders stay separate in a selection, while a re-chosen file is listed once', async () => {
  const found = await walkEntries([fakeDir('one', [fakeFile('README.md')]), fakeDir('two', [fakeFile('README.md')])]);
  assert.deepEqual(found.files.map(filePath), ['one/README.md', 'two/README.md']);
  const stamped = file => Object.assign(file, {lastModified: 5});
  const merged = mergeFiles([], found.files.map(stamped));
  assert.equal(merged.files.length, 2);
  assert.equal(merged.duplicates, 0);
  assert.equal(mergeFiles(merged.files, found.files.map(stamped)).duplicates, 2);
});

const item = (status, extra = {}) => ({item_id: 'i1', name: 'a.pdf', status, retryable: true, ...extra});

test('the ledger sorts items by what a person sees and filters them', () => {
  const items = [item('done', {output: 'a.md'}), item('error', {error: 'boom'}), item('done', {skipped: true, skip_reason: 'image_only'}), item('queued'), item('running'), item('error', {error_code: 'unsupported'})];
  assert.deepEqual(items.map(ledgerState), ['done', 'failed', 'skipped', 'queued', 'running', 'failed']);
  assert.deepEqual(ledgerCounts(items), {all: 6, done: 1, failed: 2, skipped: 1});
  assert.equal(ledgerItems(items, 'all').length, 6);
  assert.deepEqual(ledgerItems(items, 'failed').map(ledgerState), ['failed', 'failed']);
  assert.deepEqual(ledgerItems(items, 'skipped').length, 1);
  assert.deepEqual(ledgerItems(items, 'bogus').length, 6);
});

test('only an item that converting again can help is offered a retry', () => {
  assert.equal(canRetry(item('error', {error: 'HTTP 503'})), true);
  assert.equal(canRetry(item('done', {output: 'a.md'})), true);
  assert.equal(canRetry(item('done', {skipped: true, skip_reason: 'image_only'})), true);
  for (const blocked of [item('queued'), item('running'), item('error', {retryable: false}), item('done', {skip_reason: 'pending_batch', skipped: true}),
    item('error', {error_code: 'unsupported', error: "Unsupported file format: '.xyz'. Supported extensions: .pdf"}),
    item('error', {error: "Unsupported file format: '.xyz'. Supported extensions: .pdf"})]) assert.equal(canRetry(blocked), false, JSON.stringify(blocked));
  assert.equal(isUnsupported(item('error', {error_code: 'conversion_error'})), false);
  const items = [item('error', {item_id: 'a'}), item('error', {item_id: 'b', error_code: 'unsupported'}), item('done', {item_id: 'c', skipped: true}), item('error', {item_id: 'd', retryable: false}), item('error', {item_id: 'e'})];
  assert.deepEqual(failedItems(items).map(entry => entry.item_id), ['a', 'e']);
});

test('a ZIP is offered only when something produced output', () => {
  assert.equal(hasOutput([item('error'), item('done', {skipped: true, output: null})]), false);
  assert.equal(hasOutput([item('error'), item('done', {output: 'a.md'})]), true);
  assert.equal(hasOutput([]), false);
});

test('history counts converted, skipped and failed items apart and pluralizes them', () => {
  assert.deepEqual(historyParts({total: 5, done: 4, skipped: 1, failed: 1}), {total: 5, done: 3, skipped: 1, failed: 1});
  assert.deepEqual(historyParts({total: 1, done: 1}), {total: 1, done: 1, skipped: 0, failed: 0});
  assert.deepEqual(historyParts({total: 2, done: 2, skipped: 3}), {total: 2, done: 0, skipped: 3, failed: 0});
  try {
    useLocale('en');
    assert.equal(t('historyItems', {count: 1}), '1 item');
    assert.equal(t('historyItems', {count: 3}), '3 items');
    assert.equal(t('historyDone', {count: 1}), '1 done');
    assert.equal(t('historySkipped', {count: 2}), '2 skipped');
    assert.equal(t('filterFailed', {count: 4}), 'Failed 4');
    assert.equal(t('retryFailed', {count: 3}), 'Retry all failed (3)');
    assert.equal(t('retryFailedQueued', {count: 1}), '1 failed item queued again.');
    assert.equal(t('optionsChanged', {count: 1}), '1 option changed');
    assert.equal(t('llmReady', {count: 1}), 'LLM ready · 1 model available');
    assert.equal(t('selectionTruncated', {count: 1000, max: 1000}), 'Only the first 1000 files were added: one job holds at most 1000 files and URLs. Convert the rest in another job.');
    useLocale('zh');
    assert.equal(t('historyItems', {count: 1}), '共 1 项');
    assert.equal(t('filterSkipped', {count: 2}), '跳过 2');
    assert.equal(t('presetNeedsModel', {name: '标准'}), '标准 · 需要模型');
  } finally { useLocale('en'); }
});

test('the address bar names the view and keeps the open job', () => {
  assert.equal(routeView(''), 'convert');
  assert.equal(routeView('?job=abc'), 'convert');
  assert.equal(routeView('?view=history'), 'history');
  assert.equal(routeView('?view=settings&job=abc'), 'settings');
  assert.equal(routeView('?view=bogus'), 'convert');
  assert.equal(viewPath('/', '', 'history'), '/?view=history');
  assert.equal(viewPath('/', '?job=abc', 'settings'), '/?job=abc&view=settings');
  assert.equal(viewPath('/', '?view=history&job=abc', 'convert'), '/?job=abc');
  assert.equal(viewPath('/', '?view=history', 'convert'), '/');
});

test('the page offers the folder picker and every new control with its text', () => {
  const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
  assert.match(html, /<input type="file" id="folder"[^>]*webkitdirectory/);
  assert.match(html, /id="choose-folder"[^>]*data-i18n="chooseFolder"/);
  for (const id of ['llm-status', 'llm-settings-link', 'job-tools', 'ledger-filter', 'retry-failed', 'options-reset']) assert.ok(html.includes(`id="${id}"`), id);
  for (const key of ['helpPreset', 'helpProfile', 'helpStrategy', 'helpBackend']) assert.ok(html.includes(`data-i18n-title="${key}"`), key);
  // The script reads every user-visible key it names from the dictionaries.
  const app = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
  for (const key of ['filterAll', 'filterDone', 'filterFailed', 'filterSkipped']) { assert.ok(app.includes(`'${key}'`), key); assert.ok(key in DICTIONARIES.zh && key in DICTIONARIES.en, key); }
  for (const key of ['actRetryOcr', 'retryOcrTitle', 'actRetryPlain', 'retryPlainTitle', 'ledgerNone', 'retryFailedBusy', 'retryFailedPartial']) assert.ok(key in DICTIONARIES.zh && key in DICTIONARIES.en, key);
});
