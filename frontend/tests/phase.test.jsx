import { nextPhase } from '../src/components/chat/ThinkingIndicator.jsx'

let passed = 0, failed = 0
const check = (name, cond, detail = '') => {
  if (cond) { passed++; console.log('  PASS  ' + name) }
  else { failed++; console.log('  FAIL  ' + name + '   ' + detail) }
}

console.log('\n[1] Each finished step names the phase that follows it')
check('nothing yet', nextPhase(null) === 'Reading your question', nextPhase(null))
check('after condensing', nextPhase({ name: 'condense' }) === 'Choosing how to search')
check('after routing', nextPhase({ name: 'route' }) === 'Searching your documents')
check('after vector search', nextPhase({ name: 'retrieve_vector' }) === 'Ranking what it found')
check('after graph search', nextPhase({ name: 'retrieve_graph' }) === 'Ranking what it found')
check('after hybrid search', nextPhase({ name: 'retrieve_hybrid' }) === 'Ranking what it found')
check('after multi-hop', nextPhase({ name: 'retrieve_multihop' }) === 'Ranking what it found')
check('after reranking', nextPhase({ name: 'rerank' }) === 'Checking the evidence')
check('after expanding', nextPhase({ name: 'expand' }) === 'Ranking what it found')

console.log('\n[2] The one branching step is read from its own verdict')
check('sufficient -> writing',
  nextPhase({ name: 'verify', meta: { sufficient: true } }) === 'Writing the answer')
check('insufficient -> expanding',
  nextPhase({ name: 'verify', meta: { sufficient: false } }) === 'Looking for what is missing',
  nextPhase({ name: 'verify', meta: { sufficient: false } }))
check('no verdict at all does not crash',
  typeof nextPhase({ name: 'verify' }) === 'string')

console.log('\n[3] Unknown steps degrade to something harmless')
check('a step added later', nextPhase({ name: 'brand_new_node' }) === 'Working')
check('a malformed frame', typeof nextPhase({}) === 'string')

console.log('\n[4] Every step the trail can emit is covered')
const EMITTED = ['condense', 'route', 'retrieve_vector', 'retrieve_graph', 'retrieve_hybrid',
                 'retrieve_multihop', 'rerank', 'verify', 'expand', 'recover', 'synthesize',
                 'budget', 'error']
const uncovered = EMITTED.filter((n) => nextPhase({ name: n, meta: {} }) === 'Working')
check('no backend step falls through to the generic label', uncovered.length === 0, uncovered.join())

console.log('\n' + '='.repeat(60) + `\n  ${passed} passed, ${failed} failed\n` + '='.repeat(60))
process.exit(failed)
