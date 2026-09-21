import { memo, useMemo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle, Brain, User } from 'lucide-react'
import { cn } from '../../lib/cn'
import { formatDuration } from '../../lib/format'
import { CitationList } from '../citations/CitationChip'
import { ImageGallery } from '../media/ImageGallery'
import { ReasoningTrail } from '../agent/ReasoningTrail'

/**
 * Turn inline [n] markers into clickable superscripts.
 *
 * Done as a remark-free text transform inside the `p`/`li` renderers: rewriting
 * the markdown AST would be heavier, and the markers only ever appear in text
 * nodes, never inside code blocks (the model is told to cite in prose).
 */
const MARKER_RE = /\[(\d{1,2})\]/g

function renderWithMarkers(children, citations, onOpenCitation) {
  if (!citations?.length) return children

  const byIndex = new Map(citations.map((citation) => [citation.marker_index, citation]))

  const transform = (node, keyPrefix) => {
    if (typeof node !== 'string') return node
    if (!node.includes('[')) return node

    const parts = []
    let lastIndex = 0
    let match
    MARKER_RE.lastIndex = 0

    while ((match = MARKER_RE.exec(node)) !== null) {
      const index = Number(match[1])
      const citation = byIndex.get(index)
      if (!citation) continue

      if (match.index > lastIndex) parts.push(node.slice(lastIndex, match.index))
      parts.push(
        <button
          key={`${keyPrefix}-${match.index}`}
          onClick={() => onOpenCitation(citation)}
          title={`${citation.filename}${citation.page_no ? ` - page ${citation.page_no}` : ''}`}
          className={cn(
            'mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded px-1',
            'align-super text-[10px] font-semibold leading-none',
            'bg-brand/20 text-brand-soft transition-colors hover:bg-brand/35 hover:text-white'
          )}
        >
          {index}
        </button>
      )
      lastIndex = match.index + match[0].length
    }

    if (!parts.length) return node
    if (lastIndex < node.length) parts.push(node.slice(lastIndex))
    return parts
  }

  const walk = (node, keyPrefix = 'm') => {
    if (Array.isArray(node)) {
      return node.map((child, index) => walk(child, `${keyPrefix}-${index}`))
    }
    return transform(node, keyPrefix)
  }

  return walk(children)
}

function AssistantContent({ message, streaming, onOpenCitation }) {
  const citations = message.citations || []

  const components = useMemo(
    () => ({
      p: ({ children }) => <p>{renderWithMarkers(children, citations, onOpenCitation)}</p>,
      li: ({ children }) => <li>{renderWithMarkers(children, citations, onOpenCitation)}</li>,
      td: ({ children }) => <td>{renderWithMarkers(children, citations, onOpenCitation)}</td>,
      a: ({ href, children }) => (
        <a href={href} target="_blank" rel="noreferrer noopener">
          {children}
        </a>
      ),
    }),
    [citations, onOpenCitation]
  )

  return (
    <div className={cn('prose-answer', streaming && 'stream-caret')}>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {message.content || ''}
      </Markdown>
    </div>
  )
}

export const MessageBubble = memo(function MessageBubble({
  message,
  streaming = false,
  liveTraces,
  liveStatus,
  onOpenCitation,
}) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end gap-3 animate-slide-up">
        <div className="max-w-[min(42rem,85%)] rounded-xl2 rounded-br-md border border-brand/25 bg-brand/[0.12] px-4 py-2.5">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">
            {message.content}
          </p>
        </div>
        <div className="mt-0.5 h-7 w-7 shrink-0 rounded-lg border border-line bg-surface-2 p-1.5">
          <User className="h-full w-full text-ink-muted" />
        </div>
      </div>
    )
  }

  const hasError = Boolean(message.error)

  return (
    <div className="flex gap-3 animate-slide-up">
      <div
        className={cn(
          'mt-0.5 h-7 w-7 shrink-0 rounded-lg border p-1.5',
          hasError ? 'border-danger/30 bg-danger/10' : 'border-brand/25 bg-brand/10'
        )}
      >
        {hasError ? (
          <AlertTriangle className="h-full w-full text-danger" />
        ) : (
          <Brain className="h-full w-full text-brand-soft" />
        )}
      </div>

      <div className="min-w-0 max-w-[min(48rem,88%)] flex-1">
        {hasError ? (
          <div className="rounded-lg border border-danger/30 bg-danger/10 px-3.5 py-3">
            <p className="text-sm leading-relaxed text-danger">{message.content}</p>
          </div>
        ) : (
          <AssistantContent
            message={message}
            streaming={streaming}
            onOpenCitation={onOpenCitation}
          />
        )}

        {!streaming && (
          <>
            <CitationList citations={message.citations} onOpen={onOpenCitation} />
            <ImageGallery images={message.media} />
          </>
        )}

        <ReasoningTrail
          trace={streaming ? liveTraces : message.trace}
          route={streaming ? liveStatus?.route : message.route_used}
          sufficiencyScore={
            streaming ? liveStatus?.sufficiency_score : message.sufficiency_score
          }
          iterations={streaming ? liveStatus?.iterations : message.iterations}
          live={streaming}
          status={liveStatus}
        />

        {!streaming && message.latency_ms > 0 && (
          <p className="mt-1.5 text-[10px] text-ink-faint">
            Answered in {formatDuration(message.latency_ms)}
          </p>
        )}
      </div>
    </div>
  )
})
