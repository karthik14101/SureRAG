import { useEffect, useState } from 'react'
import { FileText, Hash, Image as ImageIcon, Network, Search, Sparkles, X } from 'lucide-react'
import { chatApi } from '../../api/chat'
import { cn } from '../../lib/cn'
import { truncate } from '../../lib/format'
import { Badge, Spinner } from '../ui'

const SOURCE_META = {
  vector: { label: 'Semantic match', icon: Search, tone: 'brand' },
  graph: { label: 'Graph traversal', icon: Network, tone: 'accent' },
  expand: { label: 'Expansion pass', icon: Sparkles, tone: 'warn' },
}

/** A relevance bar. Vector scores are cosine-ish; graph hits carry a flat prior. */
export function ScoreBar({ score = 0, className }) {
  const pct = Math.max(4, Math.min(100, Math.round(score * 100)))
  const tone = score >= 0.7 ? 'bg-success' : score >= 0.45 ? 'bg-warn' : 'bg-ink-faint'
  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <div className="h-1 w-14 overflow-hidden rounded-full bg-surface-3">
        <div className={cn('h-full rounded-full', tone)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] tabular-nums text-ink-faint">{score.toFixed(2)}</span>
    </div>
  )
}

/** The small numbered chip rendered under an answer. */
export function CitationChip({ citation, onOpen }) {
  const meta = SOURCE_META[citation.retrieval_source] || SOURCE_META.vector
  const Icon = meta.icon

  return (
    <button
      onClick={() => onOpen(citation)}
      title={`${citation.filename}${citation.page_no ? ` - page ${citation.page_no}` : ''}`}
      className={cn(
        'group inline-flex max-w-[15rem] items-center gap-1.5 rounded-md border border-line',
        'bg-surface-2 px-2 py-1 text-left transition-colors hover:border-brand/50 hover:bg-surface-3'
      )}
    >
      <span
        className={cn(
          'flex h-4 w-4 shrink-0 items-center justify-center rounded text-[10px] font-semibold',
          'bg-brand/20 text-brand-soft'
        )}
      >
        {citation.marker_index}
      </span>
      <span className="truncate text-[11px] text-ink-muted group-hover:text-ink">
        {citation.filename}
      </span>
      {citation.page_no != null && (
        <span className="shrink-0 text-[10px] text-ink-faint">p.{citation.page_no}</span>
      )}
      <Icon className="h-3 w-3 shrink-0 text-ink-faint" />
    </button>
  )
}

export function CitationList({ citations, onOpen }) {
  if (!citations?.length) return null
  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5">
      <span className="mr-0.5 text-[10px] font-medium uppercase tracking-wide text-ink-faint">
        Sources
      </span>
      {citations.map((citation) => (
        <CitationChip key={citation.marker_index} citation={citation} onOpen={onOpen} />
      ))}
    </div>
  )
}

/** Slide-over panel showing the cited passage in its surrounding context. */
export function CitationDrawer({ citation, onClose }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!citation?.chunk_id) {
      setDetail(null)
      return undefined
    }
    let cancelled = false
    setLoading(true)
    setError(null)

    chatApi
      .citation(citation.chunk_id)
      .then((data) => {
        if (!cancelled) {
          setDetail(data)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          // The document may have been deleted since the answer was written.
          setError(err.message)
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [citation?.chunk_id])

  useEffect(() => {
    const onKey = (event) => event.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!citation) return null

  const meta = SOURCE_META[citation.retrieval_source] || SOURCE_META.vector

  return (
    <div className="fixed inset-0 z-50 flex justify-end animate-fade-in">
      <div className="flex-1 bg-black/55 backdrop-blur-[2px]" onClick={onClose} />
      <aside className="flex w-full max-w-lg flex-col border-l border-line bg-surface shadow-2xl animate-slide-up">
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
          <div className="min-w-0 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded bg-brand/20 text-[11px] font-semibold text-brand-soft">
                {citation.marker_index}
              </span>
              <h3 className="truncate text-sm font-semibold text-ink">{citation.filename}</h3>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {citation.page_no != null && (
                <Badge>
                  <Hash className="h-2.5 w-2.5" />
                  Page {citation.page_no}
                </Badge>
              )}
              {citation.section && (
                <Badge className="max-w-[12rem]">
                  <span className="truncate">{citation.section}</span>
                </Badge>
              )}
              <Badge tone={meta.tone}>{meta.label}</Badge>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-ink-faint hover:bg-surface-2 hover:text-ink"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="border-b border-line px-5 py-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wide text-ink-faint">
              Relevance
            </span>
            <ScoreBar score={citation.score} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {loading && <Spinner label="Loading the passage..." />}

          {error && (
            <div className="space-y-3">
              <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2.5">
                <p className="text-xs leading-relaxed text-warn">
                  The full passage is no longer available (the document may have been
                  deleted). The snippet captured at answer time is shown below.
                </p>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-muted">
                {citation.snippet}
              </p>
            </div>
          )}

          {detail && (
            <div className="space-y-4">
              {detail.context_before && (
                <p className="whitespace-pre-wrap border-l-2 border-line pl-3 text-xs leading-relaxed text-ink-faint">
                  {truncate(detail.context_before, 400)}
                </p>
              )}

              <div className="rounded-lg border border-brand/25 bg-brand/[0.07] px-4 py-3.5">
                <div className="mb-2 flex items-center gap-1.5">
                  <FileText className="h-3 w-3 text-brand-soft" />
                  <span className="text-[10px] font-medium uppercase tracking-wide text-brand-soft">
                    Cited passage
                  </span>
                </div>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">
                  {detail.text}
                </p>
              </div>

              {detail.context_after && (
                <p className="whitespace-pre-wrap border-l-2 border-line pl-3 text-xs leading-relaxed text-ink-faint">
                  {truncate(detail.context_after, 400)}
                </p>
              )}

              {detail.media_ids?.length > 0 && (
                <div className="flex items-center gap-1.5 text-xs text-ink-muted">
                  <ImageIcon className="h-3.5 w-3.5" />
                  {detail.media_ids.length} image
                  {detail.media_ids.length === 1 ? '' : 's'} attached to this passage
                </div>
              )}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
