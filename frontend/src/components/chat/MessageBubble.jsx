import { memo, useMemo, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle, Brain, Trash2, User, X } from 'lucide-react'
import { cn } from '../../lib/cn'
import { formatDuration } from '../../lib/format'
import { CitationList } from '../citations/CitationChip'
import { ImageGallery } from '../media/ImageGallery'
import { ReasoningTrail } from '../agent/ReasoningTrail'
import { ThinkingIndicator } from './ThinkingIndicator'

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
            'bg-brand/20 text-brand-soft transition-colors hover:bg-brand hover:text-white'
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

/**
 * A question, with a control to remove the turn it started.
 *
 * Deletion is irreversible and this app has no modal, so the trash button asks
 * once in place rather than acting on the first click.
 */
function UserMessage({ message, deletable, onDelete }) {
  const [confirming, setConfirming] = useState(false)

  return (
    <div className="group flex justify-end gap-3 animate-slide-up">
      {deletable && (
        <div className="mt-1.5 flex items-start gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          {confirming ? (
            <>
              <button
                onClick={() => {
                  setConfirming(false)
                  onDelete?.(message.id)
                }}
                className="rounded-md px-2 py-1 text-[11px] font-medium text-danger transition-colors hover:bg-danger/10"
              >
                Delete?
              </button>
              <button
                onClick={() => setConfirming(false)}
                title="Keep it"
                aria-label="Keep this question"
                className="rounded-md p-1.5 text-ink-faint transition-colors hover:bg-surface-3 hover:text-ink"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              title="Delete this question and its answer"
              aria-label="Delete this question and its answer"
              className="rounded-md p-1.5 text-ink-faint transition-colors hover:bg-surface-3 hover:text-danger"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}

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

export const MessageBubble = memo(function MessageBubble({
  message,
  streaming = false,
  liveTraces,
  liveStatus,
  onOpenCitation,
  deletable = false,
  onDelete,
}) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <UserMessage message={message} deletable={deletable} onDelete={onDelete} />
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
        ) : streaming && !message.content ? (
          // Before the first token there is nothing to stream, and that is most
          // of the wait. Say what is happening instead of blinking a caret at
          // an empty box; the tokens replace this the moment they start.
          <ThinkingIndicator traces={liveTraces} status={liveStatus} />
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
