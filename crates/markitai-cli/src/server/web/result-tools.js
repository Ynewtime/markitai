// Comparison and printing consume only the current item's enumerated artifacts.
import {t} from './i18n.js';
export const DIFF_LIMITS = Object.freeze({characters: 2 * 1024 * 1024, lines: 6000, cells: 1_000_000});

// Like the reference preview, rendered views show only the document body;
// Source keeps the complete text including the YAML frontmatter.
export function previewBody(markdown) {
  const match = /^---\n[\s\S]*?\n---(?=\n|$)/.exec(markdown);
  return match ? markdown.slice(match[0].length) : markdown;
}

export function markdownPair(artifacts) {
  const paths = new Set(artifacts.map(value => value.relpath));
  const pairs = [...paths].filter(path => path.endsWith('.llm.md'))
    .map(enhanced => ({base: enhanced.slice(0, -7) + '.md', enhanced}))
    .filter(pair => paths.has(pair.base));
  // Ambiguous inventories must not compare unrelated documents.
  return pairs.length === 1 ? pairs[0] : null;
}

function lines(text) {
  // Keep the final newline and CRLF spelling as part of each line's identity.
  return text.match(/[^\n]*\n|[^\n]+$/g) || [];
}

export function compareLines(base, enhanced, limits = DIFF_LIMITS) {
  if (base.length > limits.characters || enhanced.length > limits.characters) return {kind: 'too-large'};
  const before = lines(base), after = lines(enhanced);
  if (before.length > limits.lines || after.length > limits.lines) return {kind: 'too-large'};
  let start = 0, suffix = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) start++;
  while (suffix < before.length - start && suffix < after.length - start && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix++;
  const n = before.length - start - suffix, m = after.length - start - suffix;
  if (n * m > limits.cells) return {kind: 'too-large'};
  const rows = before.slice(0, start).map(text => ({kind: 'same', text}));
  // A bounded table makes repeated lines deterministic without unbounded recursion.
  if (n === 0 || m === 0) {
    for (const text of before.slice(start, start + n)) rows.push({kind: 'remove', text});
    for (const text of after.slice(start, start + m)) rows.push({kind: 'add', text});
  } else {
    const width = m + 1, table = new Uint16Array((n + 1) * width);
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
      table[i * width + j] = before[start + i] === after[start + j]
        ? 1 + table[(i + 1) * width + j + 1]
        : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
    let i = 0, j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && before[start + i] === after[start + j]) { rows.push({kind: 'same', text: before[start + i]}); i++; j++; }
      else if (i < n && (j === m || table[(i + 1) * width + j] >= table[i * width + j + 1])) rows.push({kind: 'remove', text: before[start + i++]});
      else rows.push({kind: 'add', text: after[start + j++]});
    }
  }
  rows.push(...before.slice(before.length - suffix).map(text => ({kind: 'same', text})));
  return {kind: 'diff', rows, added: rows.filter(row => row.kind === 'add').length, removed: rows.filter(row => row.kind === 'remove').length};
}

export function renderComparison(comparison, target, doc = target.ownerDocument) {
  target.replaceChildren();
  if (comparison.kind === 'too-large') {
    const note = doc.createElement('p'); note.className = 'diff-note';
    note.textContent = t('diffTooLarge');
    target.append(note); return;
  }
  const summary = doc.createElement('p'); summary.className = 'diff-note';
  summary.textContent = comparison.added || comparison.removed ? t('diffSummary', {added: comparison.added, removed: comparison.removed}) : t('diffSame');
  const output = doc.createElement('div'); output.className = 'diff-lines'; output.setAttribute('aria-label', t('diffLabel'));
  let oldLine = 1, newLine = 1;
  for (const row of comparison.rows) {
    const line = doc.createElement('div'); line.className = `diff-line diff-${row.kind}`;
    const numbers = doc.createElement('span'); numbers.className = 'diff-numbers'; numbers.setAttribute('aria-hidden', 'true');
    numbers.textContent = `${row.kind === 'add' ? '' : oldLine}\t${row.kind === 'remove' ? '' : newLine}`;
    const marker = doc.createElement('span'); marker.className = 'diff-marker'; marker.textContent = row.kind === 'add' ? '+' : row.kind === 'remove' ? '−' : ' ';
    marker.setAttribute('aria-label', t(row.kind === 'add' ? 'diffAdded' : row.kind === 'remove' ? 'diffRemoved' : 'diffUnchanged'));
    const text = doc.createElement('code'); text.textContent = row.text.endsWith('\n') ? row.text.slice(0, -1) : row.text;
    line.append(numbers, marker, text);
    if (!row.text.endsWith('\n')) { const end = doc.createElement('span'); end.className = 'diff-eof'; end.textContent = t('diffNoNewline'); line.append(end); }
    output.append(line);
    if (row.kind !== 'add') oldLine++;
    if (row.kind !== 'remove') newLine++;
  }
  target.append(summary, output);
}

function aborted() { return new DOMException(t('printCancelled'), 'AbortError'); }

export function waitForPrintImages(images, {signal, timeoutMs = 10_000} = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = error => {
      if (settled) return; settled = true; clearTimeout(timer); signal?.removeEventListener('abort', cancel);
      error ? reject(error) : resolve();
    };
    const cancel = () => finish(aborted());
    const timer = setTimeout(() => finish(new Error(t('printImagesLoading'))), timeoutMs);
    signal?.addEventListener('abort', cancel, {once: true});
    if (signal?.aborted) { cancel(); return; }
    Promise.all(images.map(async image => {
      image.loading = 'eager';
      // A failed load rejects decode() with the browser's own wording; report
      // the same explicit reason as any other image that yields no pixels.
      if (!image.complete) await image.decode().catch(() => {});
      if (!image.naturalWidth) throw new Error(t('printImageBroken'));
    })).then(() => finish(), error => finish(error));
  });
}

// The caller supplies a sanitized preview, never Markdown or unsanitized HTML.
export function printPreview({preview, title, isCurrent = () => true, doc = document, win = window, timeoutMs = 10_000}) {
  const controller = new AbortController();
  const host = doc.createElement('section'); host.className = 'result-print';
  const clone = preview.cloneNode(true); clone.removeAttribute('id'); clone.removeAttribute('hidden');
  // Browser-generated PDF links must not contain authenticated asset URLs.
  for (const link of clone.querySelectorAll('a')) { link.removeAttribute('href'); link.removeAttribute('target'); link.removeAttribute('download'); }
  host.append(clone); doc.body.append(host);
  const previousTitle = doc.title;
  let restoreTitle = false, ended = false, fallback = null, finish;
  const done = new Promise((resolve, reject) => { finish = error => {
    if (ended) return; ended = true; clearTimeout(fallback); controller.abort();
    win.removeEventListener('afterprint', afterPrint); host.remove(); doc.body.classList.remove('printing-result');
    if (restoreTitle) doc.title = previousTitle;
    error ? reject(error) : resolve();
  }; });
  const afterPrint = () => finish();
  waitForPrintImages([...clone.querySelectorAll('img')], {signal: controller.signal, timeoutMs}).then(() => {
    if (ended) return;
    if (!isCurrent()) { finish(aborted()); return; }
    doc.title = title.replace(/[\u0000-\u001f\u007f]/g, '').slice(0, 160) || t('printUntitled'); restoreTitle = true;
    doc.body.classList.add('printing-result'); win.addEventListener('afterprint', afterPrint);
    // Some browsers omit afterprint when their print UI is cancelled.
    fallback = setTimeout(() => finish(), 120_000);
    try { win.print(); } catch (error) { finish(error); }
  }, error => finish(error));
  return {done, cancel: () => finish(aborted())};
}
