// Pure helpers for the Convert workspace: conversion options (public keys, help
// text, remembered choice), folder selection and the job ledger. Nothing here
// touches the DOM, so Node can test it.

// The options the service accepts. Boolean ones are tri-state in the page: unset
// keeps the server default, true and false override it.
export const OPTIONS_KEY = 'markitai.options';
export const BOOLEAN_OPTIONS = ['llm', 'ocr', 'alt', 'desc', 'screenshot', 'screenshot_only', 'pure', 'no_cache', 'no_compress'];
export const TEXT_OPTIONS = ['preset', 'profile', 'strategy', 'backend'];
// Options that only work with a model: they are switched off while none is routable.
export const LLM_OPTIONS = ['llm', 'alt', 'desc'];
const CHOICES = {
  profile: ['rag', 'obsidian', 'okf'],
  strategy: ['auto', 'static', 'playwright', 'defuddle', 'jina', 'cloudflare'],
  backend: ['native', 'cloudflare'],
};
// Dictionary key of each option's caption and of its short explanation.
export const OPTION_CAPTIONS = {llm: 'optLlm', ocr: 'optOcr', alt: 'optAlt', desc: 'optDesc', screenshot: 'optScreenshot', screenshot_only: 'optScreenshotOnly', pure: 'optPure', no_cache: 'optNoCache', no_compress: 'optNoCompress'};
export const OPTION_HELP = {
  preset: 'helpPreset', profile: 'helpProfile', strategy: 'helpStrategy', backend: 'helpBackend',
  llm: 'helpLlm', ocr: 'helpOcr', alt: 'helpAlt', desc: 'helpDesc', screenshot: 'helpScreenshot',
  screenshot_only: 'helpScreenshotOnly', pure: 'helpPure', no_cache: 'helpNoCache', no_compress: 'helpNoCompress',
};

// Only public option keys with valid values: a job's stored options can carry
// internal metadata, and a remembered choice may come from an older page.
export function publicOptions(value) {
  const clean = {};
  if (!value || typeof value !== 'object' || Array.isArray(value)) return clean;
  for (const key of BOOLEAN_OPTIONS) if (typeof value[key] === 'boolean') clean[key] = value[key];
  for (const key of TEXT_OPTIONS) {
    const text = value[key];
    if (typeof text !== 'string' || !text) continue;
    // A configured preset can have any name; the service decides whether it exists.
    if (key === 'preset' ? text.length <= 64 : CHOICES[key].includes(text)) clean[key] = text;
  }
  return clean;
}
function defaultStorage() { try { return globalThis.localStorage ?? null; } catch { return null; } }
// The last chosen options are a per-viewer convenience; a blocked or broken
// store just means the page starts from the server defaults.
export function readOptions(storage = defaultStorage()) {
  try { return publicOptions(JSON.parse(storage?.getItem(OPTIONS_KEY) ?? 'null')); } catch { return {}; }
}
export function writeOptions(value, storage = defaultStorage()) {
  try {
    const clean = publicOptions(value);
    if (Object.keys(clean).length) storage?.setItem(OPTIONS_KEY, JSON.stringify(clean)); else storage?.removeItem(OPTIONS_KEY);
  } catch { /* The choice only lasts for this page. */ }
}
// A model-dependent preset is one whose definition turns the LLM on.
export function presetNeedsModel(name, capabilities) { return capabilities?.preset_options?.[name]?.llm === true; }

// ---- Folder selection ----
const SYSTEM_FILE = /^(?:thumbs\.db|desktop\.ini)$/i;
// Hidden or system content inside a chosen folder: dot-files and dot-folders,
// Thumbs.db, desktop.ini. The folder the person chose, and files chosen one by
// one, are never filtered: that was their explicit choice.
export const isHiddenName = name => name.startsWith('.') || SYSTEM_FILE.test(name);
// The folder-relative path a File carries from a folder input (`dir/sub/a.pdf`).
export function filePath(file) { return file.markitaiPath || file.webkitRelativePath || file.name; }
function belowRoot(path) { return path.split('/').slice(1); }
// Split a folder input's files into those to keep and a count of hidden ones.
export function selectFolderFiles(values) {
  const files = [];
  let hidden = 0;
  for (const file of values) {
    const path = filePath(file);
    if (path.includes('/') && belowRoot(path).some(isHiddenName)) hidden++; else files.push(file);
  }
  return {files, hidden};
}
const readAll = reader => new Promise((resolve, reject) => {
  const all = [];
  const next = () => reader.readEntries(batch => { if (!batch.length) resolve(all); else { all.push(...batch); next(); } }, reject);
  next();
});
const entryFile = entry => new Promise((resolve, reject) => entry.file(resolve, reject));
// Walk dropped FileSystemEntry objects. Stops at `limit` files (and at a
// structural bound, so an enormous tree cannot hold the page), reporting that.
export async function walkEntries(entries, {limit = 1000, visits = 50000, depth = 64} = {}) {
  const files = [];
  let hidden = 0, truncated = false, seen = 0, unreadable = 0;
  async function visit(entry, path, level) {
    if (truncated) return;
    if (++seen > visits || level > depth) { truncated = true; return; }
    if (level > 0 && isHiddenName(entry.name)) { hidden++; return; }
    if (entry.isFile) {
      if (files.length >= limit) { truncated = true; return; }
      let file;
      try { file = await entryFile(entry); } catch { unreadable++; return; }
      Object.defineProperty(file, 'markitaiPath', {value: path, configurable: true});
      files.push(file);
    } else if (entry.isDirectory) {
      let children;
      try { children = await readAll(entry.createReader()); } catch { unreadable++; return; }
      children.sort((a, b) => a.name.localeCompare(b.name));
      for (const child of children) await visit(child, `${path}/${child.name}`, level + 1);
    }
  }
  for (const entry of entries) await visit(entry, entry.name, 0);
  return {files, hidden, truncated, unreadable};
}

// ---- Job ledger ----
export const LEDGER_FILTERS = ['all', 'done', 'failed', 'skipped'];
// What a row is, as a person would sort it: a skipped item is finished but not converted.
export function ledgerState(item) { return item.skipped ? 'skipped' : item.status === 'error' ? 'failed' : item.status; }
export function ledgerCounts(items) {
  const counts = {all: items.length, done: 0, failed: 0, skipped: 0};
  for (const item of items) { const state = ledgerState(item); if (state in counts) counts[state]++; }
  return counts;
}
export function ledgerItems(items, filter) { return filter === 'all' || !LEDGER_FILTERS.includes(filter) ? items : items.filter(item => ledgerState(item) === filter); }
// An unsupported file type cannot be fixed by converting it again.
export function isUnsupported(item) { return item.error_code === 'unsupported' || /^Unsupported file format:/.test(item.error || ''); }
export function canRetry(item) {
  return ['done', 'error'].includes(item.status) && item.retryable !== false && item.skip_reason !== 'pending_batch' && !isUnsupported(item);
}
export const failedItems = items => items.filter(item => item.status === 'error' && !item.skipped && canRetry(item));
// A finished job has something to download only if an item produced output.
export const hasOutput = items => items.some(item => item.status === 'done' && !!item.output);
// History rows count finished items; a skipped one is finished but wrote nothing.
export function historyParts(row) {
  const skipped = Math.max(0, row.skipped || 0);
  return {total: row.total || 0, done: Math.max(0, (row.done || 0) - skipped), skipped, failed: row.failed || 0};
}

// ---- Views and the address bar ----
export const VIEWS = ['convert', 'history', 'settings'];
// The view named by `?view=`; anything else is the Convert view.
export function routeView(search) { const name = new URLSearchParams(search).get('view'); return VIEWS.includes(name) ? name : 'convert'; }
// The address of a view, keeping every other query parameter (the open job).
export function viewPath(pathname, search, name) {
  const params = new URLSearchParams(search);
  if (name === 'convert') params.delete('view'); else params.set('view', name);
  const query = params.toString();
  return pathname + (query ? `?${query}` : '');
}
