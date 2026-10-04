// Keep the repository-gate entry point; pricing assertions live in one suite.
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const suite = path.join(__dirname, '../crates/markitai-cli/src/server/web/src/lib/pricing.test.ts');
const result = spawnSync(process.execPath, ['--test', suite], { stdio: 'inherit' });
if (result.error) console.error(result.error);
process.exit(result.status ?? 1);
