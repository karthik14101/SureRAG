import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Database,
  Hash,
  Image as ImageIcon,
  Search,
  SlidersHorizontal,
  X,
} from 'lucide-react'
import { kbApi } from '../../api/kb'
import { cn } from '../../lib/cn'
import { formatNumber } from '../../lib/format'
import { useToast } from '../../hooks/useToast'
import { Badge, Button, EmptyState, Spinner } from '../ui'
import { ScoreBar } from '../citations/CitationChip'

const PAGE_SIZE = 25
const COLLAPSED_CHARS = 420

const MODALITY_TONES = {
  text: 'neutral',
  table: 'accent',
  caption: 'brand',
  ocr: 'warn',
  json: 'success',
}

/**
 * Highlight query terms inside a chunk.
 *
 * Purely visual, and deliberately naive: it marks the words you typed so you can
 * see *why* a chunk came back. It is not claiming these were the matched terms —
 * dense retrieval matches meaning, not spelling, and a chunk can legitimately
 * rank highly with none of your words in it. That contrast is worth seeing.
 */
function highlight(text, query) {
  if (!query) return text
  const terms = query
    .split(/\s+/)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .filter((t) => t.length > 2)
  if (!terms.length) return text

  // One capture group means split() puts the matched separators at odd indices.
  // Parity is the test -- calling re.test() here would be wrong, because a /g
  // regex carries lastIndex between calls and would alternate its answer.
  const re = new RegExp(`(${terms.join('|')})`, 'gi')
  const parts = text.split(re)
  return parts.map((part, index) =>
    index % 2 === 1 ? (
      <mark key={index} className="rounded bg-warn/25 px-0.5 text-warn">
        {part}
      </mark>
    ) : (
      part
    )
  )
}

function ChunkCard({ chunk, query, index }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = chunk.text.length > COLLAPSED_CHARS
  const shown = expanded || !isLong ? chunk.text : `${chunk.text.slice(0, COLLAPSED_CHARS)}...`

  return (
    <article className="rounded-lg border border-line bg-surface-2/60 transition-colors hover:border-line-soft">
      <header className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2">
        <span className="flex h-5 min-w-5 items-center justify-center rounded bg-surface-3 px-1 text-[10px] font-semibold tabular-nums text-ink-faint">
          {index}
        </span>

        <span className="min-w-0 truncate text-xs font-medium text-ink" title={chunk.filename}>
          {chunk.filename}
        </span>

        {chunk.page_no != null && (
          <Badge>
            <Hash className="h-2.5 w-2.5" />
            p.{chunk.page_no}
          </Badge>
        )}

        <Badge tone={MODALITY_TONES[chunk.modality] || 'neutral'}>{chunk.modality}</Badge>

        {chunk.media_ids?.length > 0 && (
          <Badge tone="brand" title={`${chunk.media_ids.length} image(s) attached`}>
            <ImageIcon className="h-2.5 w-2.5" />
            {chunk.media_ids.length}
          </Badge>
        )}

        <span className="ml-auto flex items-center gap-2.5">
          <span
            className="text-[10px] tabular-nums text-ink-faint"
            title={`${chunk.char_count} characters, ~${chunk.token_count} tokens, position #${chunk.ordinal} in the document`}
          >
            ~{formatNumber(chunk.token_count)} tok
          </span>
          {chunk.score != null && <ScoreBar score={chunk.score} />}
        </span>
      </header>

      {chunk.section && (
        <p className="truncate border-b border-line-soft px-3 py-1.5 text-[10px] text-ink-faint">
          {chunk.section}
        </p>
      )}

      <div className="px-3 py-2.5">
        <p className="whitespace-pre-wrap break-words text-xs leading-relaxed text-ink-muted">
          {highlight(shown, query)}
        </p>
        {isLong && (
          <button
            onClick={() => setExpanded((value) => !value)}
            className="mt-1.5 inline-flex items-center gap-1 text-[10px] font-medium text-brand-soft hover:underline"
          >
            <ChevronDown className={cn('h-3 w-3 transition-transform', expanded && 'rotate-180')} />
            {expanded ? 'Show less' : `Show all ${formatNumber(chunk.char_count)} characters`}
          </button>
        )}
      </div>
    </article>
  )
}

export function ChunkExplorer({ kbId }) {
  const toast = useToast()

  const [input, setInput] = useState('')
  const [query, setQuery] = useState('')
  const [docId, setDocId] = useState('')
  const [modality, setModality] = useState('')
  const [offset, setOffset] = useState(0)

  const [result, setResult] = useState(null)
  const [facets, setFacets] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showFilters, setShowFilters] = useState(false)
  const requestId = useRef(0)

  useEffect(() => {
    kbApi.chunkFacets(kbId).then(setFacets).catch(() => {})
  }, [kbId])

  const load = useCallback(async () => {
    const id = ++requestId.current
    setLoading(true)
    try {
      const data = await kbApi.chunks(kbId, {
        q: query || undefined,
        docId: docId || undefined,
        modality: modality || undefined,
        limit: query ? 40 : PAGE_SIZE,
        offset: query ? 0 : offset,
      })
      // Ignore a stale response that arrived after a newer request.
      if (id === requestId.current) setResult(data)
    } catch (err) {
      if (id === requestId.current) toast.error(err.message || 'Could not load chunks.')
    } finally {
      if (id === requestId.current) setLoading(false)
    }
  }, [kbId, query, docId, modality, offset, toast])

  useEffect(() => {
    load()
  }, [load])

  function runSearch(event) {
    event?.preventDefault()
    setOffset(0)
    setQuery(input.trim())
  }

  function clearSearch() {
    setInput('')
    setQuery('')
    setOffset(0)
  }

  const isSearch = result?.mode === 'search'
  const total = result?.total ?? 0
  const pageStart = isSearch ? 1 : offset + 1
  const pageEnd = isSearch ? total : Math.min(offset + PAGE_SIZE, total)
  const hasFilters = Boolean(docId || modality)

  const activeDocName = useMemo(
    () => facets?.documents.find((d) => d.id === docId)?.filename,
    [facets, docId]
  )

  return (
    <div className="space-y-3 p-4">
      {/* Search bar */}
      <form onSubmit={runSearch} className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Search the index (hybrid dense + BM25)..."
            className="w-full rounded-lg border border-line bg-surface-2 py-2 pl-9 pr-8 text-xs text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
          {input && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        <Button type="submit" size="sm" disabled={!input.trim()}>
          Search
        </Button>

        <Button
          type="button"
          variant={hasFilters ? 'primary' : 'secondary'}
          size="sm"
          onClick={() => setShowFilters((open) => !open)}
        >
          <SlidersHorizontal className="h-3.5 w-3.5" />
          {hasFilters ? 'Filtered' : 'Filter'}
        </Button>
      </form>

      {/* Filters */}
      {showFilters && facets && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-surface-2 px-3 py-2.5">
          <select
            value={docId}
            onChange={(event) => {
              setDocId(event.target.value)
              setOffset(0)
            }}
            className="rounded-md border border-line bg-surface-3 px-2 py-1.5 text-[11px] text-ink focus:outline-none"
          >
            <option value="">All documents</option>
            {facets.documents.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.filename} ({doc.chunks})
              </option>
            ))}
          </select>

          <select
            value={modality}
            onChange={(event) => {
              setModality(event.target.value)
              setOffset(0)
            }}
            className="rounded-md border border-line bg-surface-3 px-2 py-1.5 text-[11px] text-ink focus:outline-none"
          >
            <option value="">All types</option>
            {facets.modalities.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name} ({item.chunks})
              </option>
            ))}
          </select>

          {hasFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setDocId('')
                setModality('')
                setOffset(0)
              }}
            >
              Clear filters
            </Button>
          )}

          <span className="ml-auto text-[10px] text-ink-faint">
            {formatNumber(facets.total_chunks)} chunks indexed
          </span>
        </div>
      )}

      {/* Status line */}
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-ink-muted">
        {isSearch ? (
          <>
            <Badge tone="brand">
              <Search className="h-2.5 w-2.5" />
              Hybrid search
            </Badge>
            <span>
              {formatNumber(total)} result{total === 1 ? '' : 's'} for{' '}
              <span className="text-ink">"{result.query}"</span>, ranked by{' '}
              {result.sparse_used ? 'RRF fusion of dense + BM25' : 'dense similarity'}
            </span>
          </>
        ) : (
          <>
            <Badge>
              <Database className="h-2.5 w-2.5" />
              Index order
            </Badge>
            <span>
              {total > 0
                ? `Showing ${formatNumber(pageStart)}-${formatNumber(pageEnd)} of ${formatNumber(total)} chunks`
                : 'No chunks'}
              {activeDocName && ` in ${activeDocName}`}
              {modality && ` (${modality})`}
            </span>
          </>
        )}
        {loading && <Spinner className="h-3 w-3" />}
      </div>

      {/* Results */}
      {!loading && !result?.items?.length && (
        <EmptyState
          icon={Database}
          title={isSearch ? 'Nothing matched' : 'No chunks yet'}
          description={
            isSearch
              ? 'Try fewer or different words, or clear the filters. Remember that dense retrieval matches meaning, so exact phrasing is not required.'
              : 'Upload and process a document to populate the index.'
          }
        />
      )}

      <div className="space-y-2">
        {result?.items?.map((chunk, index) => (
          <ChunkCard
            key={chunk.id}
            chunk={chunk}
            query={isSearch ? result.query : ''}
            index={isSearch ? index + 1 : offset + index + 1}
          />
        ))}
      </div>

      {/* Pagination (browse mode only) */}
      {!isSearch && total > PAGE_SIZE && (
        <div className="flex items-center justify-between border-t border-line pt-3">
          <Button
            variant="secondary"
            size="sm"
            disabled={offset === 0 || loading}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Previous
          </Button>

          <span className="text-[11px] tabular-nums text-ink-faint">
            Page {Math.floor(offset / PAGE_SIZE) + 1} of {Math.ceil(total / PAGE_SIZE)}
          </span>

          <Button
            variant="secondary"
            size="sm"
            disabled={offset + PAGE_SIZE >= total || loading}
            onClick={() => setOffset(offset + PAGE_SIZE)}
          >
            Next
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
    </div>
  )
}
