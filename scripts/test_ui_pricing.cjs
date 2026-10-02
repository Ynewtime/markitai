// Exercise the workbench's shipped cost labels (crates/markitai-cli/src/server/web/src/lib/pricing.ts).
// Node strips the TypeScript types itself (Node 22.18+/24); rendering is checked elsewhere.
const assert = require('node:assert/strict');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const source = path.join(__dirname, '../crates/markitai-cli/src/server/web/src/lib/pricing.ts');

(async () => {
  const {attemptPricing, priceText, attemptNotice, PRICE_WORDS} = await import(pathToFileURL(source).href);
  // The row localizes the attempt's error; an unrecognized message keeps its own text, as here.
  const message = text => ({text, detail: '', formats: ''});
  const notice = item => attemptNotice(item, PRICE_WORDS.en, message);
  let checks = 0;
  const eq = (actual, expected) => { assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected); checks++; };
  const old = {cost_usd: 0.5, error: null, pricing: {cost_status: 'complete'}};
  const usage = {requests: 1, cost_usd: 0, by_model: {legacy: {requests: 1, cost_usd: 0}}};
  const failed = {...old, diagnostics: {last_attempt: {status: 'error', error: 'Provider refused this attempt', usage}}};
  eq(notice(failed), {label: 'Last attempt failed: Price unknown · $0.000000 known subtotal', error: message('Provider refused this attempt')});
  eq(priceText(failed.cost_usd, failed.pricing), '$0.500000 · all recorded requests priced');
  eq(notice({...failed, error: 'Provider refused this attempt'}).error, null);
  eq(notice({...failed, error: 'Prior unrelated failure'}).error, message('Provider refused this attempt'));
  eq(notice({...old, diagnostics: {last_attempt: {status: 'done', error: null, usage}}}), null);
  eq(notice(old), null);
  eq(notice({cost_usd: null, error: null, diagnostics: {last_attempt: {status: 'done', error: null, usage}}}), {label: 'Last attempt: Price unknown · $0.000000 known subtotal', error: null});
  eq(notice({...old, diagnostics: {last_attempt: {status: 'error', error: '<script>literal message</script>', usage: {requests: 1, by_model: {}}}}}), {label: 'Last attempt failed', error: message('<script>literal message</script>')});
  const aggregateOnly = {requests: 0, cost_usd: 0, by_model: {sub: {requests: 0, input_tokens: 30, output_tokens: 7, priced_requests: 0, unpriced_requests: 0, cost_status: 'unknown', incomplete_request_observations: 1}}};
  eq(attemptPricing(aggregateOnly), {priced_requests: 0, unpriced_requests: 0, cost_status: 'unknown', incomplete_request_observations: 1});
  eq(priceText(0, attemptPricing(aggregateOnly)), 'Price unknown · $0.000000 known subtotal');
  const mixed = {...aggregateOnly, requests: 1, by_model: {...aggregateOnly.by_model, api: {requests: 1, priced_requests: 1, unpriced_requests: 0, cost_status: 'complete'}}};
  eq(attemptPricing(mixed), {priced_requests: 1, unpriced_requests: 0, cost_status: 'partial', incomplete_request_observations: 1});
  eq(priceText(0.5, attemptPricing(mixed)), '$0.500000 known subtotal · complete request count unavailable');
  eq(priceText(0.25, {priced_requests: 1, unpriced_requests: 2, cost_status: 'partial'}), '$0.250000 known subtotal · 2 unpriced request(s)');
  eq(priceText(0.25, undefined), '$0.250000 recorded subtotal · pricing completeness unavailable');
  eq(priceText(0, undefined), '');
  eq(priceText(0.5, {cost_status: 'complete'}, PRICE_WORDS.zh), '$0.500000 · 所有记录的请求均已计价');
  eq(attemptPricing({requests: 0, by_model: {}}), null);
  console.log(JSON.stringify({checks, status: 'passed'}));
})().catch(error => { console.error(error); process.exit(1); });
