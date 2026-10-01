'use strict';

const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'markitai-node-'));
process.env.MARKITAI_HOME = path.join(root, 'home');
const markitai = require('./index.cjs');
const source = path.join(root, '文档.md');
fs.writeFileSync(source, '# Node\n\n你好 🌍\n');
after(() => fs.rmSync(root, { recursive: true, force: true }));

test('sync and concurrent async calls preserve Unicode and result shape', async () => {
  assert.ok(markitai.version);
  const sync = markitai.convertSync(source, { config: {}, llm: false });
  assert.match(sync.markdown, /你好 🌍/);
  assert.equal(sync.output_path, null);
  const outputs = await Promise.all(Array.from({ length: 24 }, () => markitai.convert(source, { config: {}, llm: false })));
  for (const output of outputs) {
    assert.equal(output.markdown, sync.markdown);
    assert.equal(output.source, source);
    assert.equal(output.usage.requests, 0);
  }
});

test('native output files and structured errors', async () => {
  const output = await markitai.convert(source, { config: {}, output_dir: path.join(root, 'out'), llm: false });
  assert.ok(fs.existsSync(output.output_path));
  await assert.rejects(markitai.convert(path.join(root, 'missing.md'), { config: {}, llm: false }), (error) => {
    assert.ok(error instanceof markitai.ConversionError);
    assert.equal(typeof error.code, 'string');
    return true;
  });
  await assert.rejects(markitai.convert(source, { config: {}, unknown_option: true }), markitai.ConversionError);
  await assert.rejects(markitai.convert(null), TypeError);
  assert.throws(() => markitai.convertSync(source, null), TypeError);
});

test('async native fetch leaves the JavaScript event loop available', async () => {
  let requests = 0;
  const server = http.createServer((_request, response) => {
    requests++;
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end('<html><article><h1>Native HTTP</h1><p>The Node server ran.</p></article></html>');
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const url = `http://127.0.0.1:${server.address().port}/test`;
    const output = await markitai.convert(url, { config: {}, llm: false });
    assert.match(output.markdown, /The Node server ran\./);
    const cached = await markitai.convert(url, { config: {}, llm: false });
    assert.equal(cached.markdown, output.markdown);
    assert.equal(requests, 1);
    assert.equal(Object.hasOwn(cached, 'fetch_cache_hit'), false);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});

test('older producer errors and constructor preserve optional usage', async () => {
  const vm = require('node:vm');
  const loadedWrapper = Object.values(require.cache).find((entry) => entry.exports === markitai);
  assert.ok(loadedWrapper, 'test uses the actual loaded wrapper source');
  const paid = { cost_usd: 0, requests: 1, input_tokens: 0, output_tokens: 0, by_model: { fixture: { requests: 1 } } };
  for (const usage of [undefined, paid]) {
    const error = { message: 'original message', code: 'conversion_error' };
    if (usage !== undefined) error.usage = usage;
    const encoded = JSON.stringify({ ok: false, error });
    const module = { exports: {} };
    vm.runInNewContext(fs.readFileSync(loadedWrapper.filename, 'utf8'), {
      module, require: (name) => {
        assert.equal(name, './markitai.node');
        return { convertJson: async () => encoded, convertJsonSync: () => encoded, version: () => 'fixture' };
      },
    });
    const wrapper = module.exports;
    const verify = (failure) => {
      assert.ok(failure instanceof wrapper.ConversionError);
      assert.equal(failure.message, 'original message');
      assert.equal(failure.code, 'conversion_error');
      assert.equal(JSON.stringify(failure.usage), JSON.stringify(usage));
      return true;
    };
    assert.throws(() => wrapper.convertSync('unused.md'), verify);
    await assert.rejects(wrapper.convert('unused.md'), verify);
  }
  const old = new markitai.ConversionError('old message', 'old_code');
  assert.equal(old.message, 'old message');
  assert.equal(old.code, 'old_code');
  assert.equal(old.usage, undefined);
});

test('paid native failures retain separate document accounting including zero tokens', async () => {
  const files = new Map(['TERMALPHA', 'TERMBETA', 'TERMZERO'].map((name) => {
    const file = path.join(root, `${name}.md`);
    fs.writeFileSync(file, `# ${name}\n\nComplete independent source document ${name}.\n`);
    return [name, file];
  }));
  const requests = [], errors = [];
  let entered = 0, release;
  const gate = new Promise((resolve) => { release = resolve; });
  const server = http.createServer(async (request, response) => {
    try {
      let raw = '';
      for await (const chunk of request) {
        raw += chunk;
        assert.ok(raw.length < 1024 * 1024);
      }
      const payload = JSON.parse(raw);
      const text = JSON.stringify(payload.messages);
      const name = [...files.keys()].find((key) => text.includes(key));
      assert.ok(name);
      requests.push(name);
      if (name !== 'TERMZERO') {
        entered++;
        if (entered === 2) release();
        await gate;
      }
      const [input, output] = { TERMALPHA: [11, 3], TERMBETA: [29, 7], TERMZERO: [0, 0] }[name];
      response.writeHead(401, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ error: { message: 'PRIVATE RESPONSE SECRET' }, model: name, usage: { prompt_tokens: input, completion_tokens: output } }));
    } catch (error) {
      errors.push(String(error));
      response.writeHead(500).end();
    }
  });
  server.requestTimeout = 15000;
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  let gateTimedOut = false;
  const timeout = setTimeout(() => { gateTimedOut = true; release(); }, 12000);
  const config = {
    cache: { enabled: false, global_dir: path.join(root, 'paid-cache') },
    prompts: { dir: path.join(root, 'prompts') }, history: { record: false },
    ocr: { enabled: false }, image: { alt_enabled: false, desc_enabled: false },
    llm: { enabled: true, on_failure: 'fail', max_requests_per_document: 1,
      router_settings: { num_retries: 0, timeout: 15 },
      model_list: [{ model_name: 'fixture', litellm_params: { model: 'openai/fixture', api_key: 'synthetic-key', api_base: `http://127.0.0.1:${server.address().port}/v1` } }],
    },
  };
  try {
    const failure = async (name) => {
      try { await markitai.convert(files.get(name), { config }); }
      catch (error) { return error; }
      assert.fail('paid native failure unexpectedly succeeded');
    };
    const [alpha, beta] = await Promise.all([failure('TERMALPHA'), failure('TERMBETA')]);
    const zero = await failure('TERMZERO');
    for (const [name, error, input, output] of [['TERMALPHA', alpha, 11, 3], ['TERMBETA', beta, 29, 7], ['TERMZERO', zero, 0, 0]]) {
      assert.ok(error instanceof markitai.ConversionError);
      assert.equal(error.code, 'conversion_error');
      assert.match(error.message, /HTTP 401/);
      assert.ok(!error.message.includes('PRIVATE RESPONSE SECRET'));
      assert.ok(error.usage);
      assert.deepEqual([error.usage.requests, error.usage.input_tokens, error.usage.output_tokens], [1, input, output]);
      assert.deepEqual(Object.keys(error.usage.by_model), [name]);
      const coverage = error.usage.by_model[name];
      assert.deepEqual([coverage.priced_requests, coverage.unpriced_requests, coverage.cost_status], [0, 1, 'unknown']);
      assert.equal(coverage.pricing_snapshot, undefined);
      assert.equal(error.usage.cost_usd, 0);
    }
    assert.equal(entered, 2);
    assert.equal(gateTimedOut, false, 'both document requests reached the server together');
    assert.deepEqual(errors, []);
    assert.deepEqual(requests.sort(), [...files.keys()].sort());
    await assert.rejects(markitai.convert(path.join(root, 'missing-paid.md'), { config: {}, llm: false }), (error) => {
      assert.ok(error instanceof markitai.ConversionError);
      assert.equal(error.usage, undefined);
      return true;
    });
  } finally {
    clearTimeout(timeout);
    release();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});

const numbersFixtures = [
  ['test-1.numbers', 'b9e9772b2d2866c26d773fe46a173c373c7dc1dd3df6cefc7f0253b6ab50d4c3'],
  ['test-formats.numbers', '9b3ba4b52b2eb3ffd1e05602ab7da954abb2da7f37e68b147725d31777e1cda3'],
];

function numbersSource(name, sha256) {
  const directory = process.env.MARKITAI_TEST_NUMBERS_FIXTURES ||
    path.resolve(__dirname, '../../crates/markitai-core/src/formats/numbers/fixtures');
  const bytes = fs.readFileSync(path.join(directory, name));
  assert.equal(require('node:crypto').createHash('sha256').update(bytes).digest('hex'), sha256);
  return bytes;
}

function expandNumbersFixture(bytes, target) {
  // These two pinned public fixtures contain only stored ZIP entries. This is
  // test setup for their exact bytes, not a general-purpose archive extractor.
  let end = bytes.length - 22;
  while (end >= Math.max(0, bytes.length - 65557) && bytes.readUInt32LE(end) !== 0x06054b50) end--;
  assert.ok(end >= 0);
  assert.equal(end + 22 + bytes.readUInt16LE(end + 20), bytes.length);
  assert.equal(bytes.readUInt16LE(end + 4), 0);
  assert.equal(bytes.readUInt16LE(end + 6), 0);
  const count = bytes.readUInt16LE(end + 10);
  assert.equal(bytes.readUInt16LE(end + 8), count);
  let cursor = bytes.readUInt32LE(end + 16);
  const centralEnd = cursor + bytes.readUInt32LE(end + 12);
  assert.equal(centralEnd, end);
  const seen = new Set();
  fs.mkdirSync(target);
  for (let index = 0; index < count; index++) {
    assert.equal(bytes.readUInt32LE(cursor), 0x02014b50);
    assert.equal(bytes.readUInt16LE(cursor + 8), 0);
    assert.equal(bytes.readUInt16LE(cursor + 10), 0);
    const size = bytes.readUInt32LE(cursor + 24);
    assert.equal(bytes.readUInt32LE(cursor + 20), size);
    const nameSize = bytes.readUInt16LE(cursor + 28);
    const rawName = bytes.subarray(cursor + 46, cursor + 46 + nameSize);
    const name = rawName.toString('utf8');
    assert.deepEqual(Buffer.from(name, 'utf8'), rawName);
    assert.ok(!path.isAbsolute(name) && !name.includes('\\') && !name.includes('\0'));
    assert.ok(name.split('/').every((part) => part !== '..' && part !== '.'));
    assert.ok(!seen.has(name));
    seen.add(name);
    const local = bytes.readUInt32LE(cursor + 42);
    assert.equal(bytes.readUInt32LE(local), 0x04034b50);
    const data = local + 30 + bytes.readUInt16LE(local + 26) + bytes.readUInt16LE(local + 28);
    assert.ok(data + size <= bytes.readUInt32LE(end + 16));
    const destination = path.join(target, name);
    if (name.endsWith('/')) {
      fs.mkdirSync(destination, { recursive: true });
    } else {
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.writeFileSync(destination, bytes.subarray(data, data + size), { flag: 'wx', mode: 0o600 });
    }
    cursor += 46 + nameSize + bytes.readUInt16LE(cursor + 30) + bytes.readUInt16LE(cursor + 32);
  }
  assert.equal(cursor, centralEnd);
  assert.ok(count > 0);
}

function numbersOptions(directory) {
  return { config: {
    llm: { enabled: false }, ocr: { enabled: false }, screenshot: { enabled: false },
    image: { alt_enabled: false, desc_enabled: false }, cache: { enabled: false },
    history: { record: false }, prompts: { dir: path.join(directory, 'prompts') },
  }, llm: false, ocr: false, screenshot: false, alt: false, desc: false };
}

test('Numbers directory packages match pinned ZIP fixtures in sync and async calls', async () => {
  const directory = fs.mkdtempSync(path.join(root, 'numbers-'));
  const cwd = process.cwd();
  const home = process.env.HOME;
  // Load fixtures before switching to a private cwd; an installed test supplies
  // MARKITAI_TEST_NUMBERS_FIXTURES as an absolute path.
  const fixtures = numbersFixtures.map(([name, hash]) => [name, numbersSource(name, hash)]);
  process.chdir(directory);
  try {
    const options = numbersOptions(directory);
    for (const [name, bytes] of fixtures) {
      const zip = path.join(directory, name);
      const bundle = path.join(directory, `目录-${name.toUpperCase()}`);
      fs.writeFileSync(zip, bytes);
      expandNumbersFixture(bytes, bundle);
      const baseline = markitai.convertSync(zip, options);
      const sync = markitai.convertSync(bundle, options);
      const async = await markitai.convert(bundle, options);
      assert.ok(baseline.markdown.length > 0);
      for (const result of [sync, async]) {
        assert.equal(result.source, bundle);
        assert.equal(result.markdown, baseline.markdown);
        assert.deepEqual(result.warnings, baseline.warnings);
        assert.equal(result.output_path, null);
        assert.equal(result.usage.requests, 0);
        assert.deepEqual(result.assets, []);
      }
      const written = await markitai.convert(bundle, { ...options, output_dir: path.join(directory, `out-${name}`) });
      assert.equal(path.basename(written.output_path), `${path.basename(bundle)}.md`);
      assert.ok(fs.readFileSync(written.output_path, 'utf8').endsWith(baseline.markdown));
    }
    assert.equal(process.env.HOME, home);
  } finally {
    process.chdir(cwd);
  }
});

test('Numbers package support leaves ordinary directories, XML and visual modes explicit', async () => {
  const directory = fs.mkdtempSync(path.join(root, 'numbers-errors-'));
  const bytes = numbersSource(...numbersFixtures[0]);
  const bundle = path.join(directory, 'modern.numbers');
  expandNumbersFixture(bytes, bundle);
  const ordinary = path.join(directory, 'ordinary');
  const legacy = path.join(directory, 'old.numbers');
  fs.mkdirSync(ordinary);
  fs.mkdirSync(legacy);
  fs.writeFileSync(path.join(legacy, 'index.xml'), '<document><table>OLD_XML_MUST_NOT_BE_BATCHED</table></document>');
  const cwd = process.cwd();
  process.chdir(directory);
  try {
    const options = numbersOptions(directory);
    for (const [source, override, codes] of [
      [ordinary, {}, ['is_directory']],
      [legacy, {}, ['unsupported']],
      [bundle, { ocr: true }, ['unsupported']],
      [bundle, { screenshot: true }, ['unsupported']],
    ]) {
      const verify = (error) => {
        assert.ok(error instanceof markitai.ConversionError);
        assert.ok(codes.includes(error.code), `${error.code}: ${error.message}`);
        assert.equal(error.usage, undefined);
        if ((override.ocr || override.screenshot) && process.platform !== 'darwin') {
          assert.equal(error.message, 'Office page capture requires an available native PDF page renderer on this platform');
        } else if (source !== ordinary) assert.match(error.message, /Numbers/i);
        return true;
      };
      assert.throws(() => markitai.convertSync(source, { ...options, ...override }), verify);
      await assert.rejects(markitai.convert(source, { ...options, ...override }), verify);
    }
  } finally {
    process.chdir(cwd);
  }
});

// From macOS 15 (Darwin 24) dyld postpones the initialization of an image
// linked delay-initialized, and DYLD_PRINT_LIBRARIES reports each image it
// maps, postpones, and initializes later.
const postponesImages = process.platform === 'darwin' && Number(os.release().split('.')[0]) >= 24;
const mediaFrameworks = ['CoreFoundation', 'Foundation', 'CoreGraphics', 'ImageIO', 'Vision'];

function dyldImages(trace) {
  const mapped = new Set();
  const postponed = new Set();
  const initializedLater = new Set();
  for (const line of trace.split('\n')) {
    const image = line.match(/^dyld\[\d+\]: <[0-9A-F-]+> (.+)$/);
    if (image) mapped.add(path.basename(image[1]));
    const moved = line.match(/^dyld\[\d+\]: move (loaded to delayed|delayed to loaded): (.+)$/);
    if (moved) (moved[1] === 'loaded to delayed' ? postponed : initializedLater).add(moved[2]);
  }
  return { mapped, postponed, initializedLater };
}

function delayedDependencies(file) {
  // A thin 64-bit Mach-O image's dependencies whose dylib_use_command carries
  // DYLIB_USE_DELAYED_INIT.
  const image = fs.readFileSync(file);
  assert.equal(image.readUInt32LE(0), 0xfeedfacf, 'a thin 64-bit Mach-O image');
  const delayed = new Set();
  for (let index = 0, at = 32; index < image.readUInt32LE(16); index++, at += image.readUInt32LE(at + 4)) {
    const command = image.readUInt32LE(at);
    if ((command === 0xc || command === 0x80000018) && image.readUInt32LE(at + 12) === 0x1a741800
        && (image.readUInt32LE(at + 24) & 0x8) !== 0) {
      const name = image.subarray(at + image.readUInt32LE(at + 8), at + image.readUInt32LE(at + 4));
      delayed.add(path.basename(name.subarray(0, name.indexOf(0)).toString()));
    }
  }
  return delayed;
}

function tracedNode(code) {
  const run = spawnSync(process.execPath, ['-e', code], {
    env: { ...process.env, DYLD_PRINT_LIBRARIES: '1' }, encoding: 'utf8',
  });
  assert.equal(run.status, 0, run.stderr.slice(-4000));
  return run;
}

function pagePdf() {
  // One page holding a filled rectangle and no text.
  const content = '0 0.4 0.8 rg 20 20 160 60 re f';
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>',
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
  ];
  let pdf = '%PDF-1.4\n';
  const offsets = objects.map((body, index) => {
    const at = pdf.length;
    pdf += `${index + 1} 0 obj\n${body}\nendobj\n`;
    return at;
  });
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  pdf += offsets.map((at) => `${String(at).padStart(10, '0')} 00000 n \n`).join('');
  return `${pdf}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
}

test('loading the addon postpones media frameworks until a conversion needs them', {
  skip: !postponesImages && 'dyld postpones delay-initialized images from macOS 15',
}, () => {
  // The package directory of the addon this file loaded, in a checkout or an
  // installed package alike.
  const addon = Object.keys(require.cache).find((name) => path.basename(name) === 'markitai.node');
  assert.ok(addon, 'the addon is loaded');
  const delayed = delayedDependencies(addon);
  assert.deepEqual(mediaFrameworks.filter((name) => !delayed.has(name)), [], 'linked delay-initialized');
  const host = dyldImages(tracedNode('').stderr);
  if (host.mapped.size === 0) return; // This host drops dyld's diagnostic variables.
  // Frameworks that Node itself initializes at launch stay initialized.
  const expected = mediaFrameworks.filter((name) => !host.mapped.has(name) || host.postponed.has(name));
  assert.ok(expected.includes('Vision'));
  const directory = fs.mkdtempSync(path.join(root, 'frameworks-'));
  const pdf = path.join(directory, 'page.pdf');
  fs.writeFileSync(pdf, pagePdf());
  const options = {
    config: { llm: { enabled: false }, cache: { enabled: false }, history: { record: false } },
    output_dir: path.join(directory, 'out'), llm: false, ocr: false, screenshot: true, alt: false, desc: false,
  };
  const marker = 'markitai test: addon loaded';
  const run = tracedNode([
    `const markitai = require(${JSON.stringify(path.dirname(addon))});`,
    `require('node:fs').writeSync(2, ${JSON.stringify(`${marker}\n`)});`,
    `const out = markitai.convertSync(${JSON.stringify(pdf)}, ${JSON.stringify(options)});`,
    'process.stdout.write(JSON.stringify(out.screenshots));',
  ].join('\n'));
  const [loading, converting] = run.stderr.split(`${marker}\n`);
  assert.notEqual(converting, undefined, 'the marker separates loading from converting');
  const loaded = dyldImages(loading);
  assert.deepEqual(expected.filter((name) => !loaded.postponed.has(name)), []);
  assert.deepEqual([...loaded.initializedLater], [], 'loading initializes no postponed image');
  // Page rendering opens CoreGraphics on first use.
  const screenshots = JSON.parse(run.stdout);
  assert.equal(screenshots.length, 1);
  assert.deepEqual([...fs.readFileSync(screenshots[0]).subarray(0, 3)], [0xff, 0xd8, 0xff]);
  if (expected.includes('CoreGraphics')) {
    assert.ok(dyldImages(converting).initializedLater.has('CoreGraphics'));
  }
});
