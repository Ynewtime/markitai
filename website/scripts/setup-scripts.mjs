// Publishes the setup scripts as https://markitai.dev/setup.sh and /setup.ps1.
//
// Their one source is scripts/install/ at the repository root, which the
// release workflow's tests exercise; this copies them byte for byte into
// public/ (git-ignored) before every build, so the site never serves a copy
// that drifted from the tested script.
//
// Usage: bun scripts/setup-scripts.mjs   (docs:build and docs:dev run it)

import { copyFileSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..')
const SOURCE = join(SITE, '..', 'scripts', 'install')

for (const name of ['setup.sh', 'setup.ps1']) {
  const from = join(SOURCE, name)
  const to = join(SITE, 'public', name)
  try {
    copyFileSync(from, to)
  } catch (err) {
    console.error(`setup-scripts: cannot copy ${from} (${err.message.split('\n')[0]})`)
    process.exit(1)
  }
  if (!readFileSync(to).equals(readFileSync(from))) {
    console.error(`setup-scripts: public/${name} differs from scripts/install/${name}`)
    process.exit(1)
  }
  console.log(`public/${name}: copied from scripts/install/${name}`)
}
