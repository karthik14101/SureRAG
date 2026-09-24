import { ErrorBoundary } from '../src/components/ErrorBoundary.jsx'

let passed = 0, failed = 0
const check = (n, c, d = '') => { c ? (passed++, console.log('  PASS  ' + n))
                                    : (failed++, console.log('  FAIL  ' + n + '   ' + d)) }

console.log('\n[1] A thrown error becomes state, which is what stops the blank page')
const derived = ErrorBoundary.getDerivedStateFromError(new Error('boom'))
check('returns the error as state', derived && derived.error instanceof Error, JSON.stringify(derived))
check('carries the message through', derived.error.message === 'boom')

console.log('\n[2] With no error it is transparent')
const ok = new ErrorBoundary({ children: 'THE-APP' })
check('renders its children untouched', ok.render() === 'THE-APP', ok.render())

console.log('\n[3] With an error it renders a fallback instead of the children')
const broken = new ErrorBoundary({ children: 'THE-APP' })
broken.state = { error: new Error('render exploded') }
const tree = broken.render()
check('no longer returns children', tree !== 'THE-APP')
check('returns a React element', tree && typeof tree === 'object' && 'type' in tree)

const flat = JSON.stringify(tree)
check('shows the real message so it is diagnosable', flat.includes('render exploded'))
check('offers a way to recover', flat.includes('Reload') && flat.includes('Try again'))
check('reassures that data survived', flat.includes('nothing was lost'))

console.log('\n[4] Layout adapts to where it is mounted')
const root = new ErrorBoundary({ children: 'x' }); root.state = { error: new Error('e') }
const page = new ErrorBoundary({ children: 'x', compact: true }); page.state = { error: new Error('e') }
check('root fills the viewport', JSON.stringify(root.render()).includes('min-h-screen'))
check('page version fits under the header', JSON.stringify(page.render()).includes('h-full'))

console.log('\n[5] Non-Error throws do not produce "[object Object]"')
const odd = new ErrorBoundary({ children: 'x' })
odd.state = { error: 'a bare string was thrown' }
check('renders a bare string throw', JSON.stringify(odd.render()).includes('a bare string was thrown'))

console.log('\n' + '='.repeat(60) + `\n  ${passed} passed, ${failed} failed\n` + '='.repeat(60))
process.exit(failed)
