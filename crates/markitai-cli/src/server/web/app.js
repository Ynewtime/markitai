import {api, upload, uploadState, throttle, authenticatedURL, bootstrapToken, setToken, hasToken, fileURL, element, button, errorText, errorDetail, detailNode, formatSize, mergeFiles, parseUrls, copyText, MAX_FILE_BYTES} from './api.js';
import {preview} from './preview.js';
import {markdownPair, compareLines, renderComparison, printPreview} from './result-tools.js';
import {initSettings, loadSettings, renderSettings} from './settings.js';
import {t, detectLocale, setLocale, currentLocale, themePreference, setTheme, nextTheme, itemErrorMessage, persistenceMessage} from './i18n.js';
bootstrapToken();
const $ = id => document.getElementById(id);
let files = [], current = null, stream = null, poll = null, generation = 0, resultGeneration = 0;
let result = null, raw = '', documentPath = '', historyRows = [], page = 0, historyLoaded = false;
let capabilities = null, uploadController = null, lastComparison = null, offline = false, reconnectTimer = null;
let resultMode = 'preview', diffGeneration = 0, comparisonRequest = null, printSession = null, resultPending = 0;
// Static text carries data-i18n keys; dynamic labels keep their key so a language switch can redraw them.
function label(node, key, values) { node.dataset.i18n = key; node.textContent = t(key, values); }
function cancelResultTools() { diffGeneration++; lastComparison = null; comparisonRequest?.abort(); comparisonRequest = null; printSession?.cancel(); printSession = null; if(resultMode==='diff')$('comparison').textContent=t('diffPick'); }
let noticeTimer = null;
// A fixed message region stays visible wherever the action happened; confirmations fade, errors stay until dismissed.
// A translated service error keeps the service's own wording folded beside it.
function notice(message, bad = false, detail = '') {
  clearTimeout(noticeTimer);
  $('notice-text').replaceChildren(message || '');
  if (message && detail && detail !== message) $('notice-text').append(detailNode(detail));
  $('notice').hidden = !message; $('notice').classList.toggle('error', bad);
  if (message && !bad) noticeTimer = setTimeout(() => notice(''), 6000);
}
const run = async action => { try { return await action(); } catch (error) { notice(errorText(error), true, errorDetail(error)); } };
const terminal = item => ['done','error'].includes(item.status);
// Request coverage is independent of a zero or rounded cost subtotal.
function attemptPricing(usage) {
  if (!usage) return null;
  let priced = 0, unpriced = 0, incomplete = 0;
  for (const row of Object.values(usage.by_model || {})) {
    const requests = Number.isSafeInteger(row.requests) && row.requests > 0 ? row.requests : 0;
    const unknown = Number.isSafeInteger(row.incomplete_request_observations) && row.incomplete_request_observations > 0 ? row.incomplete_request_observations : 0;
    incomplete += unknown;
    if (!requests) continue;
    const known = row.priced_requests, missing = row.unpriced_requests;
    const expected = known === 0 ? 'unknown' : missing > 0 || unknown > 0 ? 'partial' : 'complete';
    if (Number.isSafeInteger(known) && known >= 0 && Number.isSafeInteger(missing) && missing >= 0 && known + missing === requests && row.cost_status === expected) {
      priced += known; unpriced += missing;
    } else unpriced += requests;
  }
  if (Number.isSafeInteger(usage.requests)) unpriced += Math.max(0, usage.requests - priced - unpriced);
  if (priced + unpriced === 0 && incomplete === 0) return null;
  const value = {priced_requests:priced, unpriced_requests:unpriced, cost_status:priced === 0 ? 'unknown' : unpriced > 0 || incomplete > 0 ? 'partial' : 'complete'};
  if (incomplete > 0) value.incomplete_request_observations = incomplete;
  return value;
}
// Cost wording is a parameter so this block stays self-contained (English by default).
const PRICE_WORDS = {
  complete: '{amount} · all recorded requests priced',
  partialIncomplete: '{amount} known subtotal · complete request count unavailable',
  partial: '{amount} known subtotal · {count} unpriced request(s)',
  unknown: 'Price unknown · {amount} known subtotal',
  recorded: '{amount} recorded subtotal · pricing completeness unavailable',
  lastFailed: 'Last attempt failed', lastFailedCost: 'Last attempt failed: {cost}', last: 'Last attempt: {cost}',
};
function fillPrice(text, values) { return text.replace(/\{(\w+)\}/g, (match, key) => key in values ? String(values[key]) : match); }
function priceText(cost, pricing, words = PRICE_WORDS) {
  if (typeof cost !== 'number' || !Number.isFinite(cost) || cost < 0) return '';
  const amount = `$${cost.toFixed(6)}`;
  if (pricing?.cost_status === 'complete') return fillPrice(words.complete, {amount});
  if (pricing?.cost_status === 'partial') return pricing.incomplete_request_observations > 0 ? fillPrice(words.partialIncomplete, {amount}) : fillPrice(words.partial, {amount, count: pricing.unpriced_requests});
  if (pricing?.cost_status === 'unknown') return fillPrice(words.unknown, {amount});
  return cost > 0 ? fillPrice(words.recorded, {amount}) : '';
}


function attemptNotice(item, words = PRICE_WORDS) {
  const attempt = item.diagnostics?.last_attempt;
  if (!attempt || (attempt.status !== 'error' && item.cost_usd != null)) return null;
  const cost = priceText(attempt.usage?.cost_usd, attemptPricing(attempt.usage), words);
  const failed = attempt.status === 'error';
  const error = failed && typeof attempt.error === 'string' && attempt.error !== item.error ? itemErrorMessage({error: attempt.error, kind: item.kind}) : null;
  return {label: failed ? (cost ? fillPrice(words.lastFailedCost, {cost}) : words.lastFailed) : cost ? fillPrice(words.last, {cost}) : '', error};
}

export async function confirmDelete(title, text) {
  const dialog = $('confirm-dialog'); $('confirm-title').textContent = title; $('confirm-message').textContent = text;
  return new Promise(resolve => { dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), {once:true}); dialog.showModal(); });
}
function priceWords() {
  return currentLocale() === 'en' ? PRICE_WORDS : {
    complete: '{amount} · 所有记录的请求均已计价', partialIncomplete: '已知小计 {amount} · 无法获得完整请求数',
    partial: '已知小计 {amount} · {count} 个请求未计价', unknown: '价格未知 · 已知小计 {amount}',
    recorded: '已记录小计 {amount} · 无法确认计价是否完整',
    lastFailed: '上次尝试失败', lastFailedCost: '上次尝试失败：{cost}', last: '上次尝试：{cost}',
  };
}
function view(name) {
  if (name !== 'convert') cancelResultTools();
  for (const section of document.querySelectorAll('.view')) section.hidden = section.id !== `${name}-view`;
  for (const link of document.querySelectorAll('[data-view]')) { const active = link.dataset.view === name; link.classList.toggle('active', active); active ? link.setAttribute('aria-current', 'page') : link.removeAttribute('aria-current'); }
  if (name === 'history') run(loadHistory);
  if (name === 'settings') run(loadSettings);
}
for (const nav of document.querySelectorAll('[data-view]')) nav.addEventListener('click', () => view(nav.dataset.view));
for (const [key, name] of Object.entries({llm:'optLlm',ocr:'optOcr',alt:'optAlt',desc:'optDesc',screenshot:'optScreenshot',screenshot_only:'optScreenshotOnly',pure:'optPure',no_cache:'optNoCache',no_compress:'optNoCompress'})) {
  const field = element('label','field'), caption = element('span'), select = element('select'); select.dataset.option = key;
  label(caption, name);
  for (const [value, text] of [['','choiceDefault'],['true','choiceOn'],['false','choiceOff']]) { const option = element('option'); option.value = value; label(option, text); select.append(option); }
  field.append(caption, select); $('boolean-options').append(field);
}
function options() {
  const value = {};
  for (const key of ['preset','profile','strategy','backend']) if ($(key).value) value[key] = $(key).value;
  for (const select of document.querySelectorAll('[data-option]')) if (select.value) value[select.dataset.option] = select.value === 'true';
  return value;
}
function addFiles(values) {
  const merged = mergeFiles(files, [...values]);
  files = merged.files; showFiles();
  if (merged.duplicates) notice(t('duplicateSkipped', {count: merged.duplicates}));
}
function showFiles(focusIndex = null) {
  $('file-list').replaceChildren();
  for (const [index, file] of files.entries()) {
    const large = file.size > MAX_FILE_BYTES;
    const row = element('li', large ? 'too-large' : '');
    const remove = button('×', () => { files.splice(index,1); showFiles(index); }, 'quiet small remove-file');
    remove.setAttribute('aria-label', t('removeFile', {name: file.name})); remove.title = remove.getAttribute('aria-label');
    row.append(element('span','filename',file.name), element('small', large ? 'item-error' : 'muted', large ? `${formatSize(file.size)} · ${t('fileTooLarge')}` : formatSize(file.size)), remove);
    $('file-list').append(row);
  }
  $('file-tools').hidden = !files.length;
  $('files-summary').textContent = files.length ? t('filesSummary', {count: files.length, size: formatSize(files.reduce((sum, file) => sum + file.size, 0))}) : '';
  $('files-clear').hidden = files.length < 2;
  // Keep keyboard users in the list after removing a row.
  if (focusIndex !== null) { const buttons = $('file-list').querySelectorAll('button'); (buttons[Math.min(focusIndex, buttons.length - 1)] || $('files')).focus(); }
}
$('files').addEventListener('change',event => { addFiles(event.target.files); event.target.value = ''; });
$('files-clear').addEventListener('click', () => { files = []; showFiles(); $('files').focus(); });
// Files dropped anywhere on the Convert view are added; elsewhere the browser must not navigate away to them.
let dragDepth = 0;
const carriesFiles = event => [...(event.dataTransfer?.types || [])].includes('Files');
function dropTarget(active) { $('drop-zone').classList.toggle('drag', active); label($('drop-title'), active ? 'dropRelease' : 'dropTitle'); }
window.addEventListener('dragenter', event => { if (!carriesFiles(event) || $('convert-view').hidden) return; dragDepth++; dropTarget(true); });
window.addEventListener('dragleave', event => { if (!carriesFiles(event)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) dropTarget(false); });
window.addEventListener('dragover', event => { if (!carriesFiles(event)) return; event.preventDefault(); event.dataTransfer.dropEffect = $('convert-view').hidden ? 'none' : 'copy'; });
window.addEventListener('drop', event => {
  if (!carriesFiles(event)) return;
  event.preventDefault(); dragDepth = 0; dropTarget(false);
  if (!$('convert-view').hidden) addFiles(event.dataTransfer.files);
});
function submitting(busy) {
  $('submit-job').disabled = busy; label($('submit-label'), busy ? 'submitBusy' : 'submit');
  $('cancel-upload').hidden = !busy || !files.length;
  // Upload progress belongs to the form; conversion progress stays in the results panel.
  $('upload-progress').hidden = !busy || !files.length;
  if (busy) showUpload(0, 0);
}
function showUpload(loaded, total) {
  const state = uploadState(loaded, total), bar = $('upload-bar');
  $('upload-progress-text').textContent = state.text;
  if (state.percent === null) bar.removeAttribute('value'); else bar.value = state.percent;
  bar.setAttribute('aria-valuetext', state.text);
  if (state.done) label($('submit-label'), 'submitCreating');
}
$('cancel-upload').addEventListener('click', () => uploadController?.abort());
$('job-stop').addEventListener('click', () => run(async () => {
  if (!current) return;
  $('job-stop').disabled = true;
  try {
    const reply = await api(`/api/jobs/${encodeURIComponent(current.job_id)}/cancel`, {method:'POST'});
    notice(t('stopRequested', {count: reply.stopping})); $('job-stop').hidden = true;
  } catch (error) {
    if (error.status === 409) { notice(t('nothingToStop')); $('job-stop').hidden = true; return; }
    throw error;
  } finally { $('job-stop').disabled = false; }
}));
$('convert-form').addEventListener('submit',async event => {
  event.preventDefault(); await run(async () => {
    const {urls, invalid} = parseUrls($('urls').value);
    if (!files.length && !urls.length && !invalid) throw new Error(t('needInput'));
    const max = capabilities?.limits?.max_job_items || 1000;
    if (files.length + urls.length > max) throw new Error(t('tooManyItems', {max}));
    const large = files.filter(file => file.size > MAX_FILE_BYTES);
    if (large.length) throw new Error(t('filesTooLarge', {names: large.map(file => file.name).join(', ')}));
    if (invalid) { $('urls').focus(); throw new Error(t(invalid.reason, {value: invalid.value})); }
    const body = new FormData(); for (const file of files) body.append('files',file,file.name);
    body.append('urls',JSON.stringify(urls)); body.append('options',JSON.stringify(options()));
    const controller = new AbortController(); uploadController = controller;
    submitting(true); notice('');
    const progress = throttle(showUpload);
    let created;
    try {
      created = await upload('/api/jobs', body, {signal: controller.signal, onProgress: progress});
    } catch (error) {
      if (error?.name === 'AbortError') { notice(t('uploadCancelled')); return; }
      throw error;
    } finally { progress.cancel(); uploadController = null; submitting(false); }
    files = []; showFiles(); $('urls').value = '';
    await openJob(created.job_id); revealJob(); $('job-title').focus({preventScroll:true});
  });
});
// On narrow screens the results sit below the form; bring them into view after an action.
function revealJob() {
  const box = $('job-title').getBoundingClientRect();
  if (box.top < 0 || box.top > innerHeight * 0.6) $('job-title').scrollIntoView({behavior:'smooth',block:'start'});
}
function closeStream() { stream?.close(); stream = null; clearTimeout(poll); poll = null; }
// A job that no longer exists must not stay in the address bar and fail on every reload.
function forgetJob(id) { const url = new URL(location.href); if (url.searchParams.get('job') !== id) return; url.searchParams.delete('job'); history.replaceState(null,'',url.pathname+url.search); }
async function openJob(id) {
  generation++; closeStream(); cancelResultTools(); resultGeneration++; resultPending=0; $('print-result').disabled=false; result = null; $('result-panel').hidden = true;
  const expected = generation;
  let snapshot;
  try { snapshot = await api(`/api/jobs/${encodeURIComponent(id)}`); }
  catch (error) { if (error.reason === 'job_not_found') forgetJob(id); throw error; }
  if (expected !== generation) return;
  current = snapshot;
  const url = new URL(location.href); url.searchParams.set('job',id); history.replaceState(null,'',url.pathname+url.search);
  view('convert'); renderJob(); if (current.status === 'running') subscribe(expected);
}
function statusText(item) {
  if (item.skip_reason === 'pending_batch') return t('statusPendingBatch');
  if (item.skipped) return t('statusSkipped');
  const key = {queued:'statusQueued',running:'statusRunning',done:'statusDone',error:'statusError'}[item.status];
  return key ? t(key) : item.status;
}
function itemProblem(item, info, expanded) {
  if (item.skip_reason === 'pending_batch') return;
  const hint = item.skipped && {image_only:'skipImageOnly', exists:'skipExists'}[item.skip_reason];
  if (hint) { info.append(element('p','item-hint',t(hint))); return; }
  if (!item.error) return;
  if (item.skipped) { info.append(element('p','item-hint',item.error)); return; }
  const problem = itemErrorMessage(item);
  if (problem.hint) { info.append(element('p','item-hint',problem.text)); return; }
  info.append(element('p','item-error',problem.text));
  problemDetail(info, problem.detail, `${item.item_id}:detail`, expanded);
  // The unsupported-format message ends with every accepted extension; keep that list folded.
  if (problem.formats) { const more = element('details','item-more'); more.dataset.key = `${item.item_id}:formats`; more.open = expanded.has(more.dataset.key); const summary = element('summary','',t('supportedFormats')); summary.dataset.key = `${more.dataset.key}:summary`; more.append(summary, element('p','',problem.formats)); info.append(more); }
}
// The service's original wording stays one click away from its translation.
function problemDetail(info, detail, key, expanded) {
  if (!detail) return;
  const more = detailNode(detail, key); more.classList.add('item-more'); more.open = expanded.has(key); info.append(more);
}
function renderJob() {
  if (!current) return;
  $('job-empty').hidden = true; $('job-progress').hidden = false;
  const completed = current.items.filter(terminal).length;
  $('progress').max = current.total || current.items.length || 1; $('progress').value = completed;
  $('job-progress-text').textContent = t('progressCount', {done: completed, total: current.items.length});
  $('job-id').textContent = current.job_id;
  const saved = current.persistence_error ? persistenceMessage(current.persistence_error) : null;
  $('job-message').textContent = saved?.text || (current.status === 'running' ? t('jobRunning') : current.failed ? t('jobAttention', {count: current.failed}) : t('jobReady'));
  if (saved?.detail) $('job-message').title = saved.detail; else $('job-message').removeAttribute('title');
  $('job-archive').hidden = current.status === 'running';
  // Only original items still waiting for a slot can be stopped; retries keep their own lifecycle.
  const waiting = current.items.filter(item => item.status === 'queued' && (item.operation || 'convert') === 'convert').length;
  $('job-stop').hidden = current.status !== 'running' || !waiting;
  $('job-archive').href = authenticatedURL(`/api/jobs/${encodeURIComponent(current.job_id)}/archive`);
  // Progress events redraw the list; keep expanded details and the focused action across redraws.
  const expanded = new Set([...$('job-items').querySelectorAll('details[open]')].map(node => node.dataset.key));
  const focused = $('job-items').contains(document.activeElement) ? document.activeElement.dataset.key : null;
  $('job-items').replaceChildren();
  const words = priceWords(), routable = !!capabilities?.llm?.routable;
  for (const item of current.items) {
    const row = element('article','job-item');
    const state = item.skipped ? 'skipped' : item.status;
    const icon = element('span',`item-icon ${state}`,state === 'done' ? '✓' : state === 'error' ? '!' : state === 'skipped' ? '–' : state === 'running' ? '↻' : '·');
    icon.setAttribute('aria-hidden','true');
    const info = element('div','item-info');
    const meta = [t(item.kind === 'url' ? 'kindUrl' : 'kindFile'), statusText(item)];
    if (item.duration_ms !== null && item.duration_ms !== undefined) meta.push(`${(item.duration_ms/1000).toFixed(1)}s`);
    if (item.llm_enhanced) meta.push(t('statusEnhanced'));
    info.append(element('strong','filename',item.name),element('small','muted',meta.join(' · ')));
    const outputCost = priceText(item.cost_usd, item.pricing, words);
    if (outputCost) info.append(element('small','muted',t('outputCost', {cost: outputCost})));
    const attempt = attemptNotice(item, words);
    if (attempt?.label) info.append(element('small','muted',attempt.label));
    if (attempt?.error) { info.append(element('p','item-error',attempt.error.text)); problemDetail(info, attempt.error.detail, `${item.item_id}:attempt`, expanded); }
    itemProblem(item, info, expanded);
    if (item.warnings?.length) { const details = element('details','warnings'); details.dataset.key = `${item.item_id}:warnings`; details.open = expanded.has(details.dataset.key); const summary = element('summary','',t('noticeCount', {count: item.warnings.length})); summary.dataset.key = `${details.dataset.key}:summary`; details.append(summary); for (const warning of item.warnings) details.append(element('p','',warning)); info.append(details); }
    const actions = element('div','item-actions');
    if (item.status === 'done' && item.output) { const view = button(t('actView'),() => run(() => openResult(item)), 'quiet small strong'); view.dataset.action = 'view'; view.setAttribute('aria-label', `${t('actView')} ${item.name}`); actions.append(view); }
    if (terminal(item) && item.retryable && item.skip_reason !== 'pending_batch') {
      const again = button(t('actRetry'),() => run(() => retry(item,false))); again.dataset.action = 'retry'; actions.append(again);
      // Enhancement needs a working model; without one the action is left out rather than shown disabled.
      if (routable) { const enhance = button(t('actEnhance'),() => run(() => retry(item,true))); enhance.title = t('enhanceTitle'); enhance.dataset.action = 'enhance'; actions.append(enhance); }
    }
    if (current.status !== 'running') actions.append(button(t('actDelete'),() => run(async () => {
      if (!await confirmDelete(t('deleteItemTitle'),t('deleteItemText', {name: item.name}))) return;
      const id = current.job_id; await api(`/api/jobs/${encodeURIComponent(id)}/items/${encodeURIComponent(item.item_id)}`,{method:'DELETE'});
      if (current.items.length === 1) { cancelResultTools(); current = null; closeStream(); $('job-items').replaceChildren(); $('job-empty').hidden = false; $('job-progress').hidden = true; $('job-archive').hidden = true; $('result-panel').hidden = true; forgetJob(id); $('job-title').focus({preventScroll:true}); }
      else await openJob(id);
    }), 'quiet small delete'));
    for (const node of actions.querySelectorAll('button')) node.dataset.key = `${item.item_id}:${node.dataset.action || 'delete'}`;
    row.append(icon,info,actions); $('job-items').append(row);
  }
  // An action that disappears (Retry while queued, a deleted row) leaves focus on the
  // same item's next control, otherwise on the results heading rather than the page.
  if (focused) {
    const nodes = [...$('job-items').querySelectorAll('button, summary')], row = `${focused.split(':')[0]}:`;
    (nodes.find(node => node.dataset.key === focused) || nodes.find(node => node.dataset.key?.startsWith(row)) || $('job-title')).focus({preventScroll:true});
  }
}
function subscribe(expected) {
  const id = current.job_id;
  stream = new EventSource(authenticatedURL(`/api/jobs/${encodeURIComponent(id)}/events`));
  for (const kind of ['snapshot','item','job']) stream.addEventListener(kind,event => {
    if (expected !== generation) return;
    try {
      const value = JSON.parse(event.data);
      if (kind === 'snapshot') current = value;
      else if (kind === 'item') { const index = current.items.findIndex(item => item.item_id === value.item_id); if (index >= 0) current.items[index] = value; }
      else Object.assign(current,value);
      renderJob();
      if (current.status !== 'running') { closeStream(); run(async () => { const value=await api(`/api/jobs/${encodeURIComponent(id)}`); if(expected===generation){current=value;renderJob();} }); }
    } catch { notice(t('progressUnreadable'),true); }
  });
  stream.onerror = () => {
    if (expected !== generation) return;
    closeStream(); $('job-message').textContent = t('progressReconnecting');
    poll = setTimeout(() => run(async () => {
      const value = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (expected !== generation) return; current = value; renderJob(); if (current.status === 'running') subscribe(expected);
    }),2000);
  };
}
async function retry(item, enhance) {
  const id = current.job_id;
  cancelResultTools();
  const body = enhance ? {operation:'enhance',options:{...current.options,llm:true}} : {operation:'retry'};
  // History options can contain internal origin metadata; only public option keys travel back.
  if (enhance) for (const key of Object.keys(body.options)) if (!['preset','llm','ocr','profile','alt','desc','screenshot','screenshot_only','pure','no_cache','no_compress','strategy','backend'].includes(key)) delete body.options[key];
  await api(`/api/jobs/${encodeURIComponent(id)}/items/${encodeURIComponent(item.item_id)}/retry`,{method:'POST',body});
  await openJob(id);
}
async function openResult(item) {
  cancelResultTools();
  const expected = ++resultGeneration, job = current.job_id;
  resultPending=expected; $('print-result').disabled=true;
  let value;
  try { value = await api(`/api/jobs/${encodeURIComponent(job)}/items/${encodeURIComponent(item.item_id)}/result`); }
  finally { if(resultPending===expected){resultPending=0;$('print-result').disabled=!!printSession;} }
  if (expected !== resultGeneration || job !== current?.job_id) return;
  result = {...value,job,pair:markdownPair(value.artifacts)}; $('result-title').textContent = item.name; $('result-panel').hidden = false;
  const variants = value.artifacts.filter(asset => asset.relpath.endsWith('.md'));
  const preferred = variants.find(asset => result.pair ? asset.relpath === result.pair[value.variant === 'llm' ? 'enhanced' : 'base'] : asset.relpath === item.output) || variants[0];
  documentPath = preferred?.relpath || item.output;
  $('result-variant').replaceChildren();
  for (const asset of variants) { const option = element('option'); label(option, (result.pair ? asset.relpath === result.pair.enhanced : value.variant === 'llm') ? 'variantEnhanced' : 'variantBase'); option.value = asset.relpath; $('result-variant').append(option); }
  $('result-variant').value = documentPath;
  // A single version needs no chooser.
  $('result-variant').hidden = variants.length < 2;
  $('artifact-list').replaceChildren();
  for (const asset of value.artifacts) { const li = element('li'); const link = element('a','',asset.relpath); link.href = authenticatedURL(fileURL(job,asset.relpath)); link.download = ''; li.append(link,element('small','muted',formatSize(asset.size))); $('artifact-list').append(li); }
  $('mode-diff').hidden = !result.pair;
  setResultMode('preview'); renderResult(value.markdown);
  $('result-panel').scrollIntoView({behavior:'smooth',block:'start'}); $('result-title').focus({preventScroll:true});
}
function renderResult(markdown) { raw = markdown; $('source').textContent = raw; $('download-source').href = authenticatedURL(fileURL(result.job,documentPath)); $('download-source').download = ''; preview(raw,result.job,documentPath,result.artifacts,$('rendered')); }
function setResultMode(mode) {
  resultMode = mode;
  for (const [name, panel] of [['preview','rendered'],['source','source'],['diff','comparison']]) {
    $(panel).hidden = name !== mode; $(`mode-${name}`).classList.toggle('active', name === mode); $(`mode-${name}`).setAttribute('aria-pressed', String(name === mode));
  }
}
async function showComparison() {
  const selected = result, pair = selected?.pair;
  if (!pair || resultPending) return;
  const expected = ++diffGeneration; comparisonRequest?.abort(); const controller = new AbortController(); comparisonRequest = controller;
  setResultMode('diff'); lastComparison = null; $('comparison').textContent = t('diffLoading');
  const currentPath = documentPath, currentText = raw;
  try {
    const limit = 8 * 1024 * 1024;
    if ([pair.base,pair.enhanced].some(path => selected.artifacts.find(asset => asset.relpath === path)?.size > limit)) {
      lastComparison = {kind:'too-large'}; renderComparison(lastComparison, $('comparison')); return;
    }
    const texts = await Promise.all([pair.base,pair.enhanced].map(path => path === currentPath ? currentText : api(fileURL(selected.job,path),{text:true,maxTextBytes:limit,signal:controller.signal})));
    if (expected !== diffGeneration || selected !== result || resultMode !== 'diff') return;
    lastComparison = compareLines(...texts); renderComparison(lastComparison, $('comparison'));
  } catch (error) {
    if (expected === diffGeneration && selected === result && error.name !== 'AbortError') $('comparison').textContent = t('diffUnavailable', {error: errorText(error)});
  } finally { if (comparisonRequest === controller) comparisonRequest = null; }
}
$('result-variant').addEventListener('change',() => run(async () => {
  cancelResultTools(); const expected=++resultGeneration, path=$('result-variant').value, selected=result;
  resultPending=expected; $('print-result').disabled=true;
  try {
    const text=await api(fileURL(selected.job,path),{text:true});
    if(expected===resultGeneration&&result===selected){documentPath=path;renderResult(text);}
  } finally {
    if(resultPending===expected){resultPending=0;$('print-result').disabled=!!printSession;$('result-variant').value=documentPath;}
  }
  if(expected===resultGeneration&&result===selected&&resultMode==='diff')await showComparison();
}));
for (const mode of ['preview','source']) $(`mode-${mode}`).addEventListener('click',() => { diffGeneration++; lastComparison = null; comparisonRequest?.abort(); setResultMode(mode); });
$('mode-diff').addEventListener('click',() => run(showComparison));
$('print-result').addEventListener('click',() => {
  if (!result || printSession || resultPending) return;
  const selected=result, path=documentPath, expected=resultGeneration;
  $('print-result').disabled=true; label($('print-result'), 'printBusy');
  const session=printPreview({preview:$('rendered'),title:path.split('/').pop().replace(/\.md$/, ''),isCurrent:()=>result===selected&&documentPath===path&&resultGeneration===expected&&!$('convert-view').hidden});
  printSession=session;
  session.done.catch(error=>{if(error.name!=='AbortError')notice(errorText(error),true);}).finally(()=>{
    if(printSession===session)printSession=null;
    if(!printSession){$('print-result').disabled=!!resultPending;label($('print-result'), 'print');}
  });
});
let copyReset = null;
$('copy-source').addEventListener('click',() => run(async () => {
  await copyText(raw);
  clearTimeout(copyReset); label($('copy-source'), 'copied'); copyReset = setTimeout(() => label($('copy-source'), 'copy'), 1800);
  notice(t('copyDone'));
}));
async function loadHistory() { historyRows=await api('/api/history');historyLoaded=true;page=0;renderHistory();$('history-archive').href=authenticatedURL('/api/history/archive'); }
function renderHistory() {
  if (!historyLoaded) return;
  const query=$('history-search').value.trim(), search=query.toLowerCase(); const rows=historyRows.filter(row=>[row.job_id,...(row.names_preview||[])].join(' ').toLowerCase().includes(search));
  const pages=Math.max(1,Math.ceil(rows.length/12));page=Math.min(page,pages-1);
  $('history-count').textContent=search?t('historyMatches',{count:rows.length,total:historyRows.length}):t('historyCount',{count:rows.length}); $('history-page').textContent=`${page+1} / ${pages}`; $('history-prev').disabled=page===0;$('history-next').disabled=page>=pages-1;
  $('history-list').parentElement.querySelector('.pagination').hidden=pages<2;
  // An empty history has no archive; the link would only open the service's JSON error.
  $('history-archive').hidden=!historyRows.length;
  $('history-list').replaceChildren();
  if(!rows.length)$('history-list').append(element('p','empty compact',search&&historyRows.length?t('historyNoMatch',{query}):t('historyEmpty')));
  const dates=new Intl.DateTimeFormat(currentLocale()==='zh'?'zh-CN':'en',{dateStyle:'medium',timeStyle:'short'});
  for(const row of rows.slice(page*12,page*12+12)){
    const names=row.names_preview||[], extra=Math.max(0,(row.total||0)-names.length);
    const title=names.join(', ')||row.job_id;
    const item=element('article','history-item');const info=element('div');
    const created=new Date(row.created_at), when=Number.isNaN(created.getTime())?row.created_at:dates.format(created);
    let meta=t('historyMeta',{date:when,total:row.total,done:row.done}); if(row.failed)meta+=` · ${t('historyFailed',{count:row.failed})}`;
    info.append(element('strong','filename',extra?`${title} ${t('historyMore',{count:extra})}`:title),element('small','muted',meta));
    const historyCost = priceText(row.cost_usd, row.pricing, priceWords()); if (historyCost) info.append(element('small','muted',historyCost));
    const actions=element('div','row');const open=button(t('actOpen'),()=>run(async()=>{await openJob(row.job_id);revealJob();$('job-title').focus({preventScroll:true});}),'quiet small strong');actions.append(open);const zip=element('a','quiet small','ZIP ↓');zip.href=authenticatedURL(`/api/jobs/${encodeURIComponent(row.job_id)}/archive`);actions.append(zip,button(t('actDelete'),()=>run(async()=>{if(await confirmDelete(t('deleteJobTitle'),t('deleteJobText',{names:title}))){await api(`/api/history/${encodeURIComponent(row.job_id)}`,{method:'DELETE'});await loadHistory();}}),'quiet small delete'));item.append(info,actions);$('history-list').append(item);
  }
}
$('history-search').addEventListener('input',()=>{page=0;renderHistory();});$('history-prev').addEventListener('click',()=>{page--;renderHistory();});$('history-next').addEventListener('click',()=>{page++;renderHistory();});$('history-refresh').addEventListener('click',()=>run(loadHistory));
function presets() {
  const selected=$('preset').value; $('preset').replaceChildren();
  for(const name of ['',...(capabilities?.presets||[])]){const option=element('option');option.value=name;const key=name?{minimal:'presetMinimal',standard:'presetStandard',rich:'presetRich'}[name]:'serverDefaults';if(key)label(option,key);else option.textContent=name[0].toUpperCase()+name.slice(1);$('preset').append(option);}
  $('preset').value=selected;
}
function serviceState(online) { offline=!online; label($('service-status'), online ? 'statusOnline' : 'statusOffline'); $('service-status').classList.toggle('offline', !online); }
// The token control matters only when this browser uses one; a 401 brings it back.
async function refreshCapabilities(){capabilities=await api('/api/capabilities');$('version').textContent=`v${capabilities.version}`;$('auth-open').hidden=!hasToken();serviceState(true);presets();renderJob();}
// While the service is unreachable, check again every few seconds and say when it is back.
window.addEventListener('markitai:offline',()=>{serviceState(false);if(reconnectTimer)return;reconnectTimer=setInterval(async()=>{try{await refreshCapabilities();}catch{return;}clearInterval(reconnectTimer);reconnectTimer=null;notice(t('reconnected'));},5000);});
window.addEventListener('markitai:online',()=>{if(offline&&capabilities)serviceState(true);});
$('notice-close').addEventListener('click',()=>notice(''));
initSettings({notice,confirmDelete,onSaved:refreshCapabilities,routable:()=>capabilities?.llm?.routable});
$('auth-open').addEventListener('click',()=>$('auth-dialog').showModal());$('auth-cancel').addEventListener('click',()=>$('auth-dialog').close());
window.addEventListener('markitai:unauthorized',()=>{$('auth-open').hidden=false;notice(t('needToken'),true);if(!$('auth-dialog').open)$('auth-dialog').showModal();});
$('auth-form').addEventListener('submit',event=>{event.preventDefault();setToken($('access-token').value);$('access-token').value='';$('auth-dialog').close();run(async()=>{await refreshCapabilities();if(current)await openJob(current.job_id);else{const id=new URL(location.href).searchParams.get('job');if(id)await openJob(id);}notice(t('connected'));});});
window.addEventListener('beforeunload',closeStream);
// Language and theme preferences; everything with a data-i18n key is relabelled in place.
for (const node of document.querySelectorAll('[data-lang]')) node.addEventListener('click',()=>setLocale(node.dataset.lang));
$('theme-toggle').addEventListener('click',()=>setTheme(nextTheme($('theme-toggle').dataset.mode||themePreference())));
window.addEventListener('markitai:locale',()=>{document.title=t('pageTitle');setTheme($('theme-toggle').dataset.mode||themePreference(),{persist:false});renderJob();renderHistory();renderSettings();if(lastComparison&&resultMode==='diff')renderComparison(lastComparison,$('comparison'));showFiles();});
setTheme(themePreference(), {persist: false});
setLocale(detectLocale(), {persist: false});
run(async()=>{await refreshCapabilities();const id=new URL(location.href).searchParams.get('job');if(id)await openJob(id);});
