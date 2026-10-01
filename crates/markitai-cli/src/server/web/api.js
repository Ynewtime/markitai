// Token handling is confined to this service; provider credentials never enter storage.
import {t} from './i18n.js';
let token = '';
const TOKEN_KEY = 'markitai.service-token';
export const MAX_FILE_BYTES = 100 * 1024 * 1024;
export function serviceURL(path, origin = location.origin) {
  const url = new URL(path, origin);
  if (url.origin !== origin || !['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('External service URL rejected');
  return url;
}
export function bootstrapToken(loc = location, hist = history, storage) {
  const url = new URL(loc.href);
  const fragment = new URLSearchParams(url.hash.slice(1));
  const supplied = fragment.get('token') ?? url.searchParams.get('token');
  if (supplied !== null) {
    token = supplied;
    fragment.delete('token'); url.searchParams.delete('token');
    url.hash = fragment.toString();
    hist.replaceState(null, '', url.pathname + url.search + url.hash);
    try { (storage ?? globalThis.sessionStorage).setItem(TOKEN_KEY, token); } catch { /* Memory remains usable when storage is unavailable. */ }
  } else {
    try { token = (storage ?? globalThis.sessionStorage).getItem(TOKEN_KEY) || ''; } catch { token = ''; }
  }
}
export function setToken(value) {
  token = value.trim();
  try { sessionStorage.setItem(TOKEN_KEY, token); } catch { /* Memory-only token. */ }
}
export function hasToken() { return token !== ''; }
export function authenticatedURL(path) {
  const url = serviceURL(path);
  if (!url.pathname.startsWith('/api/')) throw new Error('Tokens are restricted to service API URLs');
  if (token) url.searchParams.set('token', token);
  return url.href;
}
export class ApiError extends Error {
  constructor(status, body) {
    const detail = body?.detail;
    super(typeof detail === 'string' ? detail : detail?.code || t('requestFailed', {status}));
    this.status = status; this.body = body;
  }
}
// fetch rejects with a TypeError when the service cannot be reached (stopped
// server, dropped network) and for the refused redirects; neither has a status.
export class NetworkError extends Error {
  constructor() { super(t('networkError')); this.name = 'NetworkError'; }
}
function signal(name) { if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(name)); }
export async function api(path, {method = 'GET', body, signal: abort, text = false, maxTextBytes} = {}) {
  const url = serviceURL(path);
  const headers = new Headers();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (body !== undefined && !(body instanceof FormData)) {
    headers.set('Content-Type', 'application/json'); body = JSON.stringify(body);
  }
  let response;
  try { response = await fetch(url, {method, body, headers, signal: abort, credentials: 'same-origin', redirect: 'error', cache: 'no-store'}); }
  catch (error) {
    if (error?.name === 'AbortError') throw error;
    signal('markitai:offline');
    throw new NetworkError();
  }
  signal('markitai:online');
  if (!response.ok) {
    let value; try { value = await response.json(); } catch { value = null; }
    const error = new ApiError(response.status, value);
    if (response.status === 401) window.dispatchEvent(new CustomEvent('markitai:unauthorized'));
    throw error;
  }
  if (response.status === 204) return null;
  if (text && maxTextBytes !== undefined) {
    if (!Number.isSafeInteger(maxTextBytes) || maxTextBytes <= 0) throw new Error('Invalid preview size limit');
    const declared = Number(response.headers.get('content-length'));
    if (Number.isFinite(declared) && declared > maxTextBytes) { await response.body?.cancel(); throw new Error(t('diffLimit')); }
    if (!response.body) return '';
    const reader = response.body.getReader(), decoder = new TextDecoder(); let total = 0, value = '';
    try {
      for (;;) {
        const part = await reader.read(); if (part.done) break;
        total += part.value.byteLength;
        if (total > maxTextBytes) { await reader.cancel(); throw new Error(t('diffLimit')); }
        value += decoder.decode(part.value, {stream:true});
      }
      return value + decoder.decode();
    } finally { reader.releaseLock(); }
  }
  return text ? response.text() : response.json();
}
export function fileURL(job, path) {
  return `/api/jobs/${encodeURIComponent(job)}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
}
export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
export function button(text, action, className = 'quiet small') {
  const node = element('button', className, text); node.type = 'button';
  node.addEventListener('click', action); return node;
}
export function errorText(error) { return error instanceof Error ? error.message : t('requestFailedPlain'); }
export function artifactPath(target, documentPath, allowed) {
  if (/^[a-z][a-z\d+.-]*:/i.test(target) || target.startsWith('//') || target.startsWith('/') || target.includes('\\')) return null;
  let url;
  try { url = new URL(target, 'https://artifact.invalid/' + documentPath.split('/').map(encodeURIComponent).join('/')); } catch { return null; }
  let path;
  try { path = decodeURIComponent(url.pathname.slice(1)); } catch { return null; }
  return allowed.has(path) ? path : null;
}
export function formatSize(bytes) {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MiB` : `${Math.ceil(bytes / 1024)} KiB`;
}
// The same file chosen twice (name, size and modification time) is listed once.
export function mergeFiles(current, incoming) {
  const key = file => `${file.name}\u0000${file.size}\u0000${file.lastModified}`;
  const seen = new Set(current.map(key)), files = [...current];
  let duplicates = 0;
  for (const file of incoming) { if (seen.has(key(file))) { duplicates++; continue; } seen.add(key(file)); files.push(file); }
  return {files, duplicates};
}
// One address per line. A bare domain such as `example.com/page` gets https://;
// the first unusable line is reported instead of a browser parser message.
export function parseUrls(text) {
  const urls = [];
  for (const line of text.split(/\r?\n/)) {
    let value = line.trim();
    if (!value) continue;
    if (!/^[a-z][a-z\d+.-]*:/i.test(value) && /^[\w-]+(\.[\w-]+)*\.[a-z]{2,}(?:[:/?#]|$)/i.test(value)) value = `https://${value}`;
    let parsed;
    try { parsed = new URL(value); } catch { return {urls, invalid: {value: line.trim(), reason: 'badUrl'}}; }
    if (!['http:', 'https:'].includes(parsed.protocol)) return {urls, invalid: {value: line.trim(), reason: 'badScheme'}};
    if (!parsed.hostname) return {urls, invalid: {value: line.trim(), reason: 'badUrl'}};
    urls.push(value);
  }
  return {urls, invalid: null};
}
// navigator.clipboard exists only in secure contexts; a service reached over
// plain HTTP on a LAN address falls back to the selection-copy command.
export async function copyText(text, {clipboard = globalThis.navigator?.clipboard, doc = globalThis.document} = {}) {
  if (clipboard?.writeText) {
    try { await clipboard.writeText(text); return; } catch { /* Try the fallback below. */ }
  }
  const area = doc.createElement('textarea');
  area.value = text; area.setAttribute('readonly', ''); area.className = 'clipboard-buffer';
  doc.body.append(area); area.select();
  let copied = false;
  try { copied = doc.execCommand('copy'); } catch { copied = false; } finally { area.remove(); }
  if (!copied) throw new Error(t('copyFailed'));
}
