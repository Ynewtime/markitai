'use strict';

const { spawnSync } = require('node:child_process');
const { copyFileSync, mkdirSync } = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const profile = process.env.MARKITAI_BUILD_PROFILE || 'release';
if (!['debug', 'release', 'dist'].includes(profile)) {
  throw new Error('MARKITAI_BUILD_PROFILE must be debug, release, or dist');
}
// On macOS the library reads with Vision through this, beside it (docs/ocr.md). It
// is built apart, or Cargo gives it the library's portable engine, ten times its size.
const reader = process.platform === 'darwin' ? 'markitai-vision-reader' : null;
for (const name of reader ? ['markitai-node', reader] : ['markitai-node']) {
  const args = ['build', '--locked', '-p', name];
  if (profile !== 'debug') args.push('--profile', profile);
  const build = spawnSync('cargo', args, { cwd: root, stdio: 'inherit' });
  if (build.error) throw build.error;
  if (build.status !== 0) process.exit(build.status || 1);
}
const library = {
  darwin: 'libmarkitai_node.dylib',
  linux: 'libmarkitai_node.so',
  win32: 'markitai_node.dll',
}[process.platform];
if (!library) throw new Error(`Unsupported platform: ${process.platform}`);
const target = process.env.CARGO_TARGET_DIR
  ? path.resolve(root, process.env.CARGO_TARGET_DIR)
  : path.join(root, 'target');
const triple = process.env.CARGO_BUILD_TARGET;
const output = triple ? path.join(target, triple, profile) : path.join(target, profile);
mkdirSync(__dirname, { recursive: true });
copyFileSync(path.join(output, library), path.join(__dirname, 'markitai.node'));
if (reader) copyFileSync(path.join(output, reader), path.join(__dirname, reader));
