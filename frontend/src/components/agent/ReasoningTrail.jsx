import { useState } from 'react'
import {
  AlertTriangle,
  ChevronDown,
  CircleDot,
  Clock,
  Layers,
  Network,
  Repeat,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Split,
  Wand2,
} from 'lucide-react'
import { cn } from '../../lib/cn'
import { formatDuration, percent } from '../../lib/format'
import { Badge } from '../ui'

const ROUTE_META = {
  VECTOR: { label: 'Vector', icon: Search, tone: 'brand', hint: 'Semantic passage lookup' },
  GRAPH: { label: 'Graph', icon: Network, tone: 'accent', hint: 'Entity relationship traversal' },
  HYBRID: { label: 'Hybrid', icon: Layers, tone: 'brand', hint: 'Vector and graph combined' },
  MULTIHOP: { label: 'Multi-hop', icon: Split, tone: 'warn', hint: 'Decomposed into sub-questions' },
  DIRECT: { label: 'Direct', icon: Sparkles, tone: 'neutral', hint: 'Answered without retrieval' },
}

const STEP_ICONS = {
  condense: Wand2,
  route: Split,
  retrieve_vector: Search,
  retrieve_graph: Network,
  retrieve_hybrid: Layers,
  retrieve_multihop: Split,
  rerank: Layers,
  verify: ShieldCheck,
  expand: Repeat,
  recover: RotateCcw,
  synthesize: Sparkles,
  budget: Clock,
  error: AlertTriangle,
}

/** Compact route pill shown on every assistant message. */
export function RouteBadge({ route }) {
  const meta = ROUTE_META[route]
  if (!meta) return null
  const Icon = meta.icon
  return (
    <Badge tone={meta.tone} title={meta.hint}>
      <Icon className="h-2.5 w-2.5" />
      {meta.label}
    </Badge>
  )
}

const TONE_CLASS = {
  success: 'bg-success',
  warn: 'bg-warn',
  danger: 'bg-danger',
  neutral: 'bg-ink-faint',
}

// How much of the written answer carries a citation. This is the number a
// reader means by "can I trust this".
const GROUNDING = [
  { min: 0.8, tone: 'success', label: 'Grounded in your documents' },
  { min: 0.5, tone: 'warn', label: 'Partly grounded' },
  { min: 0, tone: 'danger', label: 'Largely uncited' },
]

// How much of the question the retrieved evidence covered, judged before the
// answer was written. Deliberately worded as coverage, not support: on a
// question whose answer must be assembled from several passages the verifier
// wants one that states the conclusion, and says "partly" about an answer that
// is complete.
const COVERAGE = [
  { min: 0.65, tone: 'success', label: 'Covered the question' },
  { min: 0.4, tone: 'warn', label: 'Covered in part' },
  { min: 0, tone: 'danger', label: 'Little of this in the corpus' },
]

const bucketFor = (buckets, value) => buckets.find((b) => value >= b.min) || buckets[buckets.length - 1]
const clamp = (value) => Math.max(0, Math.min(1, value))

function MeterRow({ title, value, tone, caption }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-ink-muted">{title}</span>
        <span className="text-[11px] tabular-nums text-ink">{percent(value)}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
        <div
          className={cn('h-full rounded-full transition-all', TONE_CLASS[tone])}
          style={{ width: percent(value) }}
        />
      </div>
      {caption && <p className="text-[10px] text-ink-faint">{caption}</p>}
    </div>
  )
}

/**
 * Answer confidence.
 *
 * Two different questions, kept apart because they disagree. Grounding is about
 * the answer you are reading; coverage is the SURE verifier's verdict on the
 * evidence pool before anything was written. The headline is grounding when the
 * answer has been composed, because that is what the reader is asking.
 */
export function SufficiencyMeter({
  score,
  grounding = null,
  iterations = 0,
  outOfScope = false,
  compact = false,
}) {
  if (score == null) return null

  const coverage = clamp(score)
  const hasGrounding = grounding != null
  const headline = hasGrounding ? clamp(grounding) : coverage
  const headlineBucket = bucketFor(hasGrounding ? GROUNDING : COVERAGE, headline)

  if (compact) {
    if (outOfScope) {
      return (
        <span
          className="text-[10px] text-warn"
          title="This knowledge base does not hold the kind of material this question needs."
        >
          Not in this KB
        </span>
      )
    }
    return (
      <span
        className="inline-flex items-center gap-1.5"
        title={`${headlineBucket.label} — ${
          hasGrounding ? 'share of the answer that cites a source' : 'evidence coverage'
        } ${percent(headline)}`}
      >
        <span className="h-1 w-8 overflow-hidden rounded-full bg-surface-3">
          <span
            className={cn('block h-full rounded-full', TONE_CLASS[headlineBucket.tone])}
            style={{ width: percent(headline) }}
          />
        </span>
        <span className="text-[10px] tabular-nums text-ink-faint">{percent(headline)}</span>
      </span>
    )
  }

  const coverageBucket = bucketFor(COVERAGE, coverage)
  const passes =
    iterations > 0 ? ` · ${iterations} expansion pass${iterations === 1 ? '' : 'es'} run to fill gaps` : ''

  return (
    <div className="space-y-2.5">
      {outOfScope && (
        <p className="flex items-start gap-1.5 text-[10px] text-warn">
          <AlertTriangle className="mt-px h-2.5 w-2.5 shrink-0" />
          This knowledge base does not hold the kind of material this question needs, so the
          answer says what is absent rather than inferring it.
        </p>
      )}
      {hasGrounding && (
        <MeterRow
          title="Answer grounding"
          value={headline}
          tone={headlineBucket.tone}
          caption={`${headlineBucket.label} · share of the answer's claims that cite a source`}
        />
      )}
      <MeterRow
        title="Evidence coverage"
        value={coverage}
        tone={coverageBucket.tone}
        caption={`${coverageBucket.label}${passes}`}
      />
    </div>
  )
}

/** The expandable step-by-step log of how the answer was produced. */
export function ReasoningTrail({ trace, route, sufficiencyScore, iterations, live = false, status }) {
  const [open, setOpen] = useState(false)
  const steps = trace || []

  if (!steps.length && !live) return null

  const totalMs = steps.reduce((sum, step) => sum + (step.duration_ms || 0), 0)
  // A DIRECT turn answers without documents, so there is no evidence to score.
  // Showing "0% - thin evidence" there reads as a failure rather than a choice.
  const showSufficiency = sufficiencyScore != null && route !== 'DIRECT'
  // Grounding and the scope verdict travel in the synthesis step's metadata
  // rather than their own columns, so older messages simply lack them.
  const synthesis = steps.find((step) => step.name === 'synthesize')
  const grounding = synthesis?.meta?.grounding ?? null
  const outOfScope = Boolean(synthesis?.meta?.out_of_scope)

  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-line bg-surface-2/60">
      <button
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-surface-2"
      >
        <ChevronDown
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-ink-faint transition-transform',
            open && 'rotate-180'
          )}
        />
        <span className="text-[11px] font-medium text-ink-muted">
          {live ? 'Working...' : 'How this answer was built'}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {route && <RouteBadge route={route} />}
          {showSufficiency && (
            <SufficiencyMeter
              score={sufficiencyScore}
              grounding={grounding}
              iterations={iterations}
              outOfScope={outOfScope}
              compact
            />
          )}
          {!live && totalMs > 0 && (
            <span className="text-[10px] tabular-nums text-ink-faint">
              {formatDuration(totalMs)}
            </span>
          )}
        </div>
      </button>

      {open && (
        <div className="space-y-2.5 border-t border-line px-3 py-3">
          {showSufficiency && (
            <div className="pb-1">
              <SufficiencyMeter
                score={sufficiencyScore}
                grounding={grounding}
                iterations={iterations}
                outOfScope={outOfScope}
              />
            </div>
          )}

          <ol className="space-y-2">
            {steps.map((step, index) => {
              const Icon = STEP_ICONS[step.name] || CircleDot
              const isError = step.name === 'error'
              const isVerify = step.name === 'verify'
              return (
                <li key={`${step.name}-${index}`} className="flex gap-2.5">
                  <div className="flex flex-col items-center">
                    <div
                      className={cn(
                        'rounded-md border p-1',
                        isError
                          ? 'border-danger/30 bg-danger/10'
                          : isVerify
                            ? 'border-brand/30 bg-brand/10'
                            : 'border-line bg-surface-3'
                      )}
                    >
                      <Icon
                        className={cn(
                          'h-3 w-3',
                          isError ? 'text-danger' : isVerify ? 'text-brand-soft' : 'text-ink-muted'
                        )}
                      />
                    </div>
                    {index < steps.length - 1 && <div className="mt-1 w-px flex-1 bg-line" />}
                  </div>

                  <div className="min-w-0 flex-1 pb-1">
                    <div className="flex items-baseline gap-2">
                      <span
                        className={cn(
                          'text-[11px] font-medium',
                          isError ? 'text-danger' : 'text-ink'
                        )}
                      >
                        {step.label}
                      </span>
                      {step.duration_ms > 0 && (
                        <span className="text-[10px] tabular-nums text-ink-faint">
                          {formatDuration(step.duration_ms)}
                        </span>
                      )}
                    </div>
                    {step.detail && (
                      <p className="mt-0.5 text-[11px] leading-relaxed text-ink-muted">
                        {step.detail}
                      </p>
                    )}
                    {step.meta?.entities?.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {step.meta.entities.slice(0, 6).map((entity) => (
                          <Badge key={entity} tone="accent">
                            {entity}
                          </Badge>
                        ))}
                      </div>
                    )}
                    {step.meta?.sub_questions?.length > 0 && (
                      <ul className="mt-1 space-y-0.5">
                        {step.meta.sub_questions.map((question) => (
                          <li key={question} className="text-[10px] text-ink-faint">
                            → {question}
                          </li>
                        ))}
                      </ul>
                    )}
                    {step.meta?.corpus && (
                      <p className="mt-1 text-[10px] leading-relaxed text-ink-faint">
                        <span className="text-ink-muted">judged against:</span>{' '}
                        {step.meta.corpus}
                      </p>
                    )}
                    {step.meta?.missing?.length > 0 && (
                      <ul className="mt-1 space-y-0.5">
                        {step.meta.missing.slice(0, 3).map((gap) => (
                          <li key={gap} className="text-[10px] text-warn">
                            missing: {gap}
                          </li>
                        ))}
                      </ul>
                    )}
                    {step.meta?.queries?.length > 0 && (
                      <ul className="mt-1 space-y-0.5">
                        {step.meta.queries.map((query) => (
                          <li key={query} className="text-[10px] text-ink-faint">
                            searched: "{query}"
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </li>
              )
            })}

            {live && (
              <li className="flex items-center gap-2.5 pl-1">
                <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-brand" />
                <span className="text-[11px] text-ink-faint">
                  {status?.route ? `Retrieving via ${status.route}...` : 'Thinking...'}
                </span>
              </li>
            )}
          </ol>

          {status?.warnings?.length > 0 && (
            <div className="space-y-1 border-t border-line pt-2">
              {status.warnings.map((warning) => (
                <p key={warning} className="flex items-start gap-1.5 text-[10px] text-warn">
                  <AlertTriangle className="mt-px h-2.5 w-2.5 shrink-0" />
                  {warning}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
