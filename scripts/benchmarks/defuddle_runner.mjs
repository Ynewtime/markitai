// Run the built reference implementation offline, including DOM and Markdown.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
const [root, manifestPath, iterationsText] = process.argv.slice(2);
const require = createRequire(resolve(root, 'package.json'));
const { Defuddle } = require(resolve(root, 'dist/node.js'));
const { parseLinkedomHTML } = require(resolve(root, 'dist/utils/linkedom-compat.js'));
const offlineFetch = async () => { throw new Error('Network disabled in comparison'); };
const extract = (html, url) => Defuddle(parseLinkedomHTML(html, url), url, {
  separateMarkdown: true, fetch: offlineFetch,
});
await extract('<article><p>Warm up the extraction pipeline.</p></article>', 'https://example.com');
const results = [];
for (const item of JSON.parse(readFileSync(manifestPath, 'utf8'))) {
  const html = readFileSync(item.path, 'utf8');
  const times = [];
  try {
    let result;
    for (let n = 0; n < Number(iterationsText); n++) {
      const start = performance.now();
      result = await extract(html, item.url);
      times.push(performance.now() - start);
    }
    results.push({ name: item.name, milliseconds: times,
      markdown: result.contentMarkdown ?? '', title: result.title,
      author: result.author, published: result.published, word_count: result.wordCount });
  } catch (error) {
    results.push({ name: item.name, milliseconds: times, error: String(error) });
  }
}
process.stdout.write(JSON.stringify(results));
