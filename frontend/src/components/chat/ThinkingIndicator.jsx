import { useEffect, useState } from 'react'

/**
 * What the engine is doing, while it is doing it.
 *
 * Roughly two thirds of a turn happens before the first token exists: routing,
 * retrieval, reranking and verification all run first, and on this corpus that
 * is fifteen to twenty-six seconds of nothing to stream. The steps were already
 * arriving live over SSE the whole time -- they were just folded inside a
 * collapsed accordion, so the entire wait read as a single static "Working...".
 * Showing what the pipeline is doing rather than merely that it is working is
 * the difference between a progress bar and a spinner.
 */

// Each trace frame arrives when its step FINISHES, so the newest one names what
// just completed, not what is running. The next phase follows from it, and the
// one branch that is not fixed -- whether a verifier sends the loop back for
// another pass -- is answered by that step's own verdict.
const NEXT_PHASE = {
  condense: 'Choosing how to search',
  route: 'Searching your documents',
  retrieve_vector: 'Ranking what it found',
  retrieve_graph: 'Ranking what it found',
  retrieve_hybrid: 'Ranking what it found',
  retrieve_multihop: 'Ranking what it found',
  rerank: 'Checking the evidence',
  expand: 'Ranking what it found',
  recover: 'Writing the answer',
  budget: 'Writing the answer',
  synthesize: 'Writing the answer',
  // A failed turn ends the stream a moment later, but the frame can land first.
  error: 'Something went wrong',
}

export function nextPhase(step) {
  if (!step) return 'Reading your question'
  if (step.name === 'verify') {
    return step.meta?.sufficient ? 'Writing the answer' : 'Looking for what is missing'
  }
  return NEXT_PHASE[step.name] || 'Working'
}

/** Ticks once a second while mounted. Restarts whenever the turn does. */
function useElapsed() {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    const started = Date.now()
    const id = setInterval(() => {
      setSeconds(Math.floor((Date.now() - started) / 1000))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  return seconds
}

export function ThinkingIndicator({ traces = [], status }) {
  const seconds = useElapsed()
  const last = traces.length ? traces[traces.length - 1] : null

  const phase = nextPhase(last)
  // The step that just finished already says something concrete -- "Retrieved
  // 12 passage(s) from 1 document(s)" -- which beats any wording invented here.
  const detail = last?.detail

  return (
    <div className="py-0.5">
      <div className="flex items-center gap-2.5">
        <span className="h-1.5 w-1.5 shrink-0 animate-pulse-soft rounded-full bg-brand" />
        <span className="text-sm text-ink-muted">
          {phase}
          <span className="animate-pulse-soft">...</span>
        </span>
        {/* Removes the only question a spinner cannot answer: is it stuck? */}
        <span className="ml-auto text-[11px] tabular-nums text-ink-faint">{seconds}s</span>
      </div>

      {detail && (
        <p className="mt-1.5 pl-4 text-[11px] leading-relaxed text-ink-faint">{detail}</p>
      )}

      {!detail && status?.route && (
        <p className="mt-1.5 pl-4 text-[11px] text-ink-faint">
          Searching via {status.route.toLowerCase()}
        </p>
      )}
    </div>
  )
}
