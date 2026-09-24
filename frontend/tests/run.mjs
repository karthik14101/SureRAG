/**
 * Run the frontend tests.
 *
 *     npm test
 *
 * There is no test framework here on purpose. These tests cover pure logic --
 * the phase mapping and the error boundary's contract -- which needs a bundler
 * to resolve JSX and nothing else. esbuild already ships with Vite, so the
 * suite runs with no extra dependency and no config to keep in step.
 *
 * If this grows to need rendering, a real runner (vitest) is the right answer
 * rather than building one here.
 */
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = mkdtempSync(join(tmpdir(), 'sure-tests-'))

const files = readdirSync(HERE).filter((name) => name.endsWith('.test.jsx'))
if (!files.length) {
  console.error('No .test.jsx files found in', HERE)
  process.exit(1)
}

let failed = 0

for (const file of files) {
  const bundle = join(OUT, file.replace('.test.jsx', '.cjs'))
  try {
    execFileSync(
      'npx',
      [
        'esbuild', join(HERE, file),
        '--bundle', '--platform=node', '--format=cjs',
        // Vite uses the automatic JSX runtime; the classic one emits
        // React.createElement against an import these files do not have.
        '--jsx=automatic',
        '--loader:.jsx=jsx',
        `--outfile=${bundle}`,
        '--log-level=error',
      ],
      { stdio: 'inherit', shell: process.platform === 'win32' }
    )
    execFileSync('node', [bundle], { stdio: 'inherit' })
  } catch {
    failed += 1
  }
}

process.exit(failed ? 1 : 0)
