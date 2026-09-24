import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  FileText,
  Network,
  Plus,
  Quote,
  Search,
  Share2,
  X,
} from 'lucide-react'
import { kbApi } from '../../api/kb'
import { cn } from '../../lib/cn'
import { formatNumber, truncate } from '../../lib/format'
import { predicateLabel, typeColor, typeLabel } from '../../lib/graphColors'
import { useToast } from '../../hooks/useToast'
import { Badge, Button, EmptyState, Modal, Spinner } from '../ui'
import { GraphCanvas } from './GraphCanvas'

/**
 * Browse the knowledge graph: the entities and relations the LLM extracted
 * during ingestion, and the chunks each one came from.
 *
 * Beyond being interesting to look at, this is the only way to see how good the
 * extraction actually was. Duplicate entities ("President" vs "President of
 * India"), junk entities and vague predicates all weaken GRAPH and MULTIHOP
 * answers, and all of them are obvious here.
 */

const ENTITY_PAGE = 30
const RELATION_PAGE = 50
const OVERVIEW_NODES = 50
const MAX_CANVAS_NODES = 180

const SORTS = [
  ['degree', 'Most connected'],
  ['mentions', 'Most mentioned'],
  ['name', 'A–Z'],
]

/** Debounce a fast-changing value (search boxes) before it drives a request. */
function useDebounced(value, delay = 300) {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return settled
}

function TypeDot({ type, className }) {
  return (
    <span
      className={cn('h-2 w-2 shrink-0 rounded-full', className)}
      style={{ backgroundColor: typeColor(type) }}
      title={typeLabel(type)}
    />
  )
}

/** Clickable entity reference used inside relation rows. */
function EntityChip({ entity, onOpen }) {
  return (
    <button
      onClick={() => onOpen(entity.id)}
      className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-line bg-surface-3 px-1.5 py-0.5 text-[11px] text-ink transition-colors hover:border-brand/50 hover:text-brand-soft"
      title={`${entity.name} (${typeLabel(entity.type)})`}
    >
      <TypeDot type={entity.type} />
      <span className="truncate">{entity.name}</span>
    </button>
  )
}

function SegmentedControl({ options, value, onChange }) {
  return (
    <div className="flex items-center gap-0.5 rounded-md border border-line bg-surface-2 p-0.5">
      {options.map(([key, label]) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={cn(
            'rounded px-2 py-1 text-[11px] transition-colors',
            value === key ? 'bg-surface-3 font-medium text-ink' : 'text-ink-muted hover:text-ink'
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Chunk preview                                                               */
/* -------------------------------------------------------------------------- */
function highlightTerm(text, term) {
  if (!term || !text) return text
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  // One capture group -> the matches land on odd indices after split().
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
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

function ChunkPreview({ chunkId, term, onClose }) {
  const [chunk, setChunk] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!chunkId) return undefined
    let cancelled = false
    setChunk(null)
    setError('')
    kbApi
      .chunk(chunkId)
      .then((data) => !cancelled && setChunk(data))
      .catch((err) => !cancelled && setError(err.message || 'Could not load that passage.'))
    return () => {
      cancelled = true
    }
  }, [chunkId])

  return (
    <Modal
      open={Boolean(chunkId)}
      onClose={onClose}
      title="Source passage"
      description={
        chunk ? `${chunk.filename}${chunk.page_no != null ? ` · page ${chunk.page_no}` : ''}` : ' '
      }
      width="max-w-2xl"
    >
      {error && <p className="text-xs text-danger">{error}</p>}
      {!chunk && !error && <Spinner label="Loading passage..." />}
      {chunk && (
        <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
          {chunk.section && <p className="text-[10px] text-ink-faint">{chunk.section}</p>}
          {chunk.context_before && (
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink-faint">
              {truncate(chunk.context_before, 280)}
            </p>
          )}
          <p className="whitespace-pre-wrap rounded-md border border-line bg-surface-2 p-3 text-xs leading-relaxed text-ink-muted">
            {highlightTerm(chunk.text, term)}
          </p>
          {chunk.context_after && (
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink-faint">
              {truncate(chunk.context_after, 280)}
            </p>
          )}
        </div>
      )}
    </Modal>
  )
}

/* -------------------------------------------------------------------------- */
/* Entity list (left column)                                                   */
/* -------------------------------------------------------------------------- */
function EntityList({ kbId, facets, selectedId, onOpen }) {
  const toast = useToast()
  const [input, setInput] = useState('')
  const query = useDebounced(input)
  const [type, setType] = useState('')
  const [sort, setSort] = useState('degree')
  const [offset, setOffset] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const requestId = useRef(0)

  useEffect(() => setOffset(0), [query, type, sort])

  useEffect(() => {
    const id = ++requestId.current
    setLoading(true)
    kbApi
      .graphEntities(kbId, { q: query || undefined, type: type || undefined, sort, limit: ENTITY_PAGE, offset })
      .then((data) => id === requestId.current && setResult(data))
      .catch((err) => id === requestId.current && toast.error(err.message || 'Could not load entities.'))
      .finally(() => id === requestId.current && setLoading(false))
  }, [kbId, query, type, sort, offset, toast])

  const total = result?.total ?? 0

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Search entities..."
          className="w-full rounded-lg border border-line bg-surface-2 py-2 pl-8 pr-7 text-xs text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/40"
        />
        {input && (
          <button
            onClick={() => setInput('')}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink"
            aria-label="Clear search"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      <div className="flex gap-1.5">
        <select
          value={type}
          onChange={(event) => setType(event.target.value)}
          className="min-w-0 flex-1 rounded-md border border-line bg-surface-3 px-2 py-1.5 text-[11px] text-ink focus:outline-none"
        >
          <option value="">All types</option>
          {facets?.types.map((item) => (
            <option key={item.name} value={item.name}>
              {typeLabel(item.name)} ({item.count})
            </option>
          ))}
        </select>
        <select
          value={sort}
          onChange={(event) => setSort(event.target.value)}
          className="min-w-0 flex-1 rounded-md border border-line bg-surface-3 px-2 py-1.5 text-[11px] text-ink focus:outline-none"
        >
          {SORTS.map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2 text-[10px] text-ink-faint">
        <span>
          {formatNumber(total)} {total === 1 ? 'entity' : 'entities'}
          {query && ' matched'}
        </span>
        {loading && <Spinner className="h-3 w-3" />}
      </div>

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {result?.items.map((entity) => (
          <button
            key={entity.id}
            onClick={() => onOpen(entity.id)}
            className={cn(
              'w-full rounded-md border px-2 py-1.5 text-left transition-colors',
              entity.id === selectedId
                ? 'border-brand/50 bg-brand/10'
                : 'border-line bg-surface-2/60 hover:border-line-soft hover:bg-surface-2'
            )}
          >
            <span className="flex items-center gap-1.5">
              <TypeDot type={entity.type} />
              <span className="min-w-0 flex-1 truncate text-xs text-ink" title={entity.name}>
                {entity.name}
              </span>
              <span
                className="shrink-0 text-[10px] tabular-nums text-ink-faint"
                title={`${entity.degree} relation(s), mentioned in ${entity.mentions} chunk(s)`}
              >
                {entity.degree}
              </span>
            </span>
          </button>
        ))}

        {!loading && !result?.items.length && (
          <p className="px-2 py-6 text-center text-[11px] text-ink-faint">
            {query ? 'No entity matches that.' : 'No entities.'}
          </p>
        )}
      </div>

      {total > ENTITY_PAGE && (
        <div className="flex items-center justify-between border-t border-line pt-2">
          <Button
            variant="ghost"
            size="sm"
            disabled={offset === 0 || loading}
            onClick={() => setOffset(Math.max(0, offset - ENTITY_PAGE))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-[10px] tabular-nums text-ink-faint">
            {Math.floor(offset / ENTITY_PAGE) + 1} / {Math.ceil(total / ENTITY_PAGE)}
          </span>
          <Button
            variant="ghost"
            size="sm"
            disabled={offset + ENTITY_PAGE >= total || loading}
            onClick={() => setOffset(offset + ENTITY_PAGE)}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Entity detail                                                               */
/* -------------------------------------------------------------------------- */
function RelationRow({ relation, direction, onOpen, onEvidence }) {
  const other =
    direction === 'out'
      ? { id: relation.target_id, name: relation.target_name, type: relation.target_type }
      : { id: relation.source_id, name: relation.source_name, type: relation.source_type }

  return (
    <li className="flex flex-wrap items-center gap-1.5 py-1">
      <ArrowRight
        className={cn('h-3 w-3 shrink-0 text-ink-faint', direction === 'in' && 'rotate-180')}
      />
      <span className="font-mono text-[10px] uppercase tracking-tight text-accent">
        {predicateLabel(relation.predicate)}
      </span>
      <EntityChip entity={other} onOpen={onOpen} />
      {relation.weight > 1 && (
        <span className="text-[10px] text-ink-faint" title="Extracted from this many chunks">
          ×{relation.weight}
        </span>
      )}
      {relation.chunk_ids?.length > 0 && (
        <button
          onClick={() => onEvidence(relation.chunk_ids[0])}
          className="text-ink-faint transition-colors hover:text-brand-soft"
          title="Show the passage this came from"
                      aria-label="Show the passage this came from"
        >
          <Quote className="h-3 w-3" />
        </button>
      )}
    </li>
  )
}

function RelationGroup({ title, relations, direction, onOpen, onEvidence }) {
  const [expanded, setExpanded] = useState(false)
  if (!relations.length) return null
  const shown = expanded ? relations : relations.slice(0, 10)

  return (
    <div>
      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-ink-faint">
        {title} ({relations.length})
      </p>
      <ul className="divide-y divide-line-soft">
        {shown.map((relation) => (
          <RelationRow
            key={relation.id}
            relation={relation}
            direction={direction}
            onOpen={onOpen}
            onEvidence={onEvidence}
          />
        ))}
      </ul>
      {relations.length > 10 && (
        <button
          onClick={() => setExpanded((value) => !value)}
          className="mt-1 text-[10px] font-medium text-brand-soft hover:underline"
        >
          {expanded ? 'Show fewer' : `Show all ${relations.length}`}
        </button>
      )}
    </div>
  )
}

function EntityDetail({ detail, loading, onOpen, onFocus, onExpand, onEvidence }) {
  if (loading && !detail) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-line bg-surface-2/50 py-10">
        <Spinner label="Loading entity..." />
      </div>
    )
  }

  if (!detail) {
    return (
      <div className="rounded-lg border border-dashed border-line bg-surface-2/30 px-4 py-8 text-center">
        <p className="text-xs text-ink-muted">
          Select an entity to see its relationships and the passages it came from.
        </p>
        <p className="mt-1 text-[10px] text-ink-faint">
          Click a node to inspect it · double-click a node to pull in its neighbours
        </p>
      </div>
    )
  }

  const { entity, outgoing, incoming, mentions } = detail

  return (
    <div className="space-y-3 rounded-lg border border-line bg-surface-2/50 p-3">
      <div className="flex flex-wrap items-start gap-2">
        <TypeDot type={entity.type} className="mt-1.5" />
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-ink">{entity.name}</h3>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-ink-faint">
            <span>{typeLabel(entity.type)}</span>
            <span>·</span>
            <span>
              {formatNumber(entity.degree)} relation{entity.degree === 1 ? '' : 's'}
            </span>
            <span>·</span>
            <span>
              {formatNumber(entity.mentions)} mention{entity.mentions === 1 ? '' : 's'}
            </span>
          </p>
        </div>
        <div className="flex gap-1.5">
          <Button variant="secondary" size="sm" onClick={() => onFocus(entity.id)} title="Redraw the graph around this entity">
            <Crosshair className="h-3.5 w-3.5" />
            Focus
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onExpand(entity.id)} title="Add its neighbours to the graph">
            <Plus className="h-3.5 w-3.5" />
            Expand
          </Button>
        </div>
      </div>

      {entity.description && (
        <p className="text-xs leading-relaxed text-ink-muted">{entity.description}</p>
      )}

      {!outgoing.length && !incoming.length && (
        <p className="text-[11px] text-ink-faint">
          No relationships were extracted for this entity — it appears in the text but the model
          found nothing to link it to.
        </p>
      )}

      <RelationGroup title="Outgoing" relations={outgoing} direction="out" onOpen={onOpen} onEvidence={onEvidence} />
      <RelationGroup title="Incoming" relations={incoming} direction="in" onOpen={onOpen} onEvidence={onEvidence} />

      {mentions.length > 0 && (
        <div>
          <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-ink-faint">
            Found in {mentions.length} passage{mentions.length === 1 ? '' : 's'}
          </p>
          <div className="space-y-1">
            {mentions.slice(0, 8).map((mention) => (
              <button
                key={mention.chunk_id}
                onClick={() => onEvidence(mention.chunk_id)}
                className="w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-left transition-colors hover:border-brand/40"
              >
                <span className="flex items-center gap-1.5 text-[10px] text-ink-faint">
                  <FileText className="h-3 w-3" />
                  <span className="min-w-0 truncate">{mention.filename}</span>
                  {mention.page_no != null && <span>· p.{mention.page_no}</span>}
                  {mention.count > 1 && <span>· ×{mention.count}</span>}
                </span>
                {mention.snippet && (
                  <span className="mt-0.5 block text-[11px] leading-relaxed text-ink-muted">
                    {truncate(mention.snippet, 150)}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Relations table                                                             */
/* -------------------------------------------------------------------------- */
function RelationsView({ kbId, facets, onOpen, onEvidence }) {
  const toast = useToast()
  const [input, setInput] = useState('')
  const query = useDebounced(input)
  const [predicate, setPredicate] = useState('')
  const [offset, setOffset] = useState(0)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const requestId = useRef(0)

  useEffect(() => setOffset(0), [query, predicate])

  useEffect(() => {
    const id = ++requestId.current
    setLoading(true)
    kbApi
      .graphRelations(kbId, {
        q: query || undefined,
        predicate: predicate || undefined,
        limit: RELATION_PAGE,
        offset,
      })
      .then((data) => id === requestId.current && setResult(data))
      .catch((err) => id === requestId.current && toast.error(err.message || 'Could not load relations.'))
      .finally(() => id === requestId.current && setLoading(false))
  }, [kbId, query, predicate, offset, toast])

  const total = result?.total ?? 0

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Search by entity or predicate..."
            className="w-full rounded-lg border border-line bg-surface-2 py-2 pl-9 pr-3 text-xs text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/40"
          />
        </div>
        <select
          value={predicate}
          onChange={(event) => setPredicate(event.target.value)}
          className="rounded-md border border-line bg-surface-3 px-2 py-1.5 text-[11px] text-ink focus:outline-none"
        >
          <option value="">All predicates</option>
          {facets?.predicates.map((item) => (
            <option key={item.name} value={item.name}>
              {predicateLabel(item.name)} ({item.count})
            </option>
          ))}
        </select>
        <span className="flex items-center gap-2 text-[11px] text-ink-faint">
          {formatNumber(total)} relation{total === 1 ? '' : 's'}
          {loading && <Spinner className="h-3 w-3" />}
        </span>
      </div>

      <div className="overflow-hidden rounded-lg border border-line">
        <table className="w-full table-fixed">
          <thead className="bg-surface-2 text-[10px] uppercase tracking-wide text-ink-faint">
            <tr>
              <th className="w-[32%] px-3 py-2 text-left font-medium">Source</th>
              <th className="w-[24%] px-3 py-2 text-left font-medium">Relationship</th>
              <th className="w-[32%] px-3 py-2 text-left font-medium">Target</th>
              <th className="w-[12%] px-3 py-2 text-right font-medium">Evidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {result?.items.map((relation) => (
              <tr key={relation.id} className="align-middle hover:bg-surface-2/50">
                <td className="px-3 py-1.5">
                  <EntityChip
                    entity={{
                      id: relation.source_id,
                      name: relation.source_name,
                      type: relation.source_type,
                    }}
                    onOpen={onOpen}
                  />
                </td>
                <td className="px-3 py-1.5">
                  <span
                    className="block truncate font-mono text-[10px] uppercase text-accent"
                    title={relation.description || relation.predicate}
                  >
                    {predicateLabel(relation.predicate)}
                  </span>
                </td>
                <td className="px-3 py-1.5">
                  <EntityChip
                    entity={{
                      id: relation.target_id,
                      name: relation.target_name,
                      type: relation.target_type,
                    }}
                    onOpen={onOpen}
                  />
                </td>
                <td className="px-3 py-1.5 text-right">
                  <span className="inline-flex items-center gap-2">
                    {relation.weight > 1 && (
                      <span className="text-[10px] text-ink-faint">×{relation.weight}</span>
                    )}
                    {relation.chunk_ids?.length > 0 && (
                      <button
                        onClick={() => onEvidence(relation.chunk_ids[0])}
                        className="text-ink-faint transition-colors hover:text-brand-soft"
                        title="Show the passage this came from"
                      aria-label="Show the passage this came from"
                      >
                        <Quote className="h-3 w-3" />
                      </button>
                    )}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {!loading && !result?.items.length && (
          <p className="py-8 text-center text-xs text-ink-faint">Nothing matched.</p>
        )}
      </div>

      {total > RELATION_PAGE && (
        <div className="flex items-center justify-between">
          <Button
            variant="secondary"
            size="sm"
            disabled={offset === 0 || loading}
            onClick={() => setOffset(Math.max(0, offset - RELATION_PAGE))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Previous
          </Button>
          <span className="text-[11px] tabular-nums text-ink-faint">
            Page {Math.floor(offset / RELATION_PAGE) + 1} of {Math.ceil(total / RELATION_PAGE)}
          </span>
          <Button
            variant="secondary"
            size="sm"
            disabled={offset + RELATION_PAGE >= total || loading}
            onClick={() => setOffset(offset + RELATION_PAGE)}
          >
            Next
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Explorer                                                                    */
/* -------------------------------------------------------------------------- */
const EMPTY_GRAPH = { key: 'empty', nodes: [], edges: [], anchor: null }

export function GraphExplorer({ kbId }) {
  const toast = useToast()

  const [facets, setFacets] = useState(null)
  const [loadingFacets, setLoadingFacets] = useState(true)
  const [view, setView] = useState('explore')

  const [graph, setGraph] = useState(EMPTY_GRAPH)
  const [graphLoading, setGraphLoading] = useState(true)
  const [centerId, setCenterId] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [hops, setHops] = useState(1)
  const [layout, setLayout] = useState('force')

  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [evidenceChunk, setEvidenceChunk] = useState(null)
  const detailRequest = useRef(0)

  useEffect(() => {
    setLoadingFacets(true)
    kbApi
      .graphFacets(kbId)
      .then(setFacets)
      .catch((err) => toast.error(err.message || 'Could not load the graph.'))
      .finally(() => setLoadingFacets(false))
  }, [kbId, toast])

  const showOverview = useCallback(async () => {
    setGraphLoading(true)
    try {
      const data = await kbApi.graphOverview(kbId, OVERVIEW_NODES)
      setGraph({ key: `overview:${Date.now()}`, nodes: data.nodes, edges: data.edges, anchor: null })
      setCenterId(null)
    } catch (err) {
      toast.error(err.message || 'Could not load the graph.')
    } finally {
      setGraphLoading(false)
    }
  }, [kbId, toast])

  useEffect(() => {
    showOverview()
  }, [showOverview])

  const loadDetail = useCallback(
    async (entityId) => {
      const id = ++detailRequest.current
      setDetailLoading(true)
      // Drop the previous entity's details straight away, so the panel never
      // shows one entity's relations under another's name.
      setDetail(null)
      try {
        const data = await kbApi.graphEntity(kbId, entityId)
        if (id === detailRequest.current) setDetail(data)
      } catch (err) {
        if (id === detailRequest.current) toast.error(err.message || 'Could not load that entity.')
      } finally {
        if (id === detailRequest.current) setDetailLoading(false)
      }
    },
    [kbId, toast]
  )

  /** Select a node: show its details, but leave the drawing as it is. */
  const select = useCallback(
    (entityId) => {
      setSelectedId(entityId)
      loadDetail(entityId)
    },
    [loadDetail]
  )

  /** Redraw the graph around an entity. */
  const focus = useCallback(
    async (entityId, nextHops = hops) => {
      setView('explore')
      setSelectedId(entityId)
      loadDetail(entityId)
      setGraphLoading(true)
      try {
        const data = await kbApi.graphNeighbourhood(kbId, entityId, {
          hops: nextHops,
          limit: nextHops > 1 ? 70 : 40,
        })
        setGraph({
          key: `focus:${entityId}:${nextHops}:${Date.now()}`,
          nodes: data.nodes,
          edges: data.edges,
          anchor: null,
        })
        setCenterId(entityId)
      } catch (err) {
        toast.error(err.message || 'Could not load that neighbourhood.')
      } finally {
        setGraphLoading(false)
      }
    },
    [kbId, hops, loadDetail, toast]
  )

  /** Add an entity's neighbours to what is already drawn. */
  const expand = useCallback(
    async (entityId) => {
      setSelectedId(entityId)
      loadDetail(entityId)
      try {
        const data = await kbApi.graphNeighbourhood(kbId, entityId, { hops: 1, limit: 25 })
        setGraph((previous) => {
          const nodes = new Map(previous.nodes.map((node) => [node.id, node]))
          if (nodes.size >= MAX_CANVAS_NODES) {
            toast.info('The graph is getting crowded — use Focus to start a fresh view.')
            return previous
          }
          const anchorDepth = nodes.get(entityId)?.depth ?? 0
          for (const node of data.nodes) {
            if (!nodes.has(node.id)) nodes.set(node.id, { ...node, depth: anchorDepth + 1 })
          }
          const edges = new Map(previous.edges.map((edge) => [edge.id, edge]))
          for (const edge of data.edges) edges.set(edge.id, edge)
          return {
            key: previous.key,
            nodes: [...nodes.values()],
            edges: [...edges.values()],
            anchor: entityId,
          }
        })
      } catch (err) {
        toast.error(err.message || 'Could not expand that entity.')
      }
    },
    [kbId, loadDetail, toast]
  )

  function changeHops(next) {
    setHops(next)
    if (centerId) focus(centerId, next)
  }

  // The centre node carries its true relation count, so the status line can say
  // how much of a busy entity's neighbourhood is actually on screen.
  const center = useMemo(
    () => graph.nodes.find((node) => node.id === centerId),
    [graph.nodes, centerId]
  )
  const drawnNeighbours = graph.nodes.length - 1
  const hiddenNeighbours = center ? Math.max(0, center.degree - drawnNeighbours) : 0

  if (loadingFacets) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner label="Loading the knowledge graph..." />
      </div>
    )
  }

  if (facets && !facets.available) {
    return (
      <EmptyState
        icon={Network}
        title="Knowledge graph offline"
        description="Neo4j is not reachable, so there are no entities to browse. Start it with `docker compose up -d`, then refresh this page. Vector search works without it."
      />
    )
  }

  if (facets && facets.total_entities === 0) {
    return (
      <EmptyState
        icon={Network}
        title="No entities extracted yet"
        description="Entities and relationships are extracted while documents are ingested, when GRAPH_EXTRACTION is `selective` or `on`. If your documents are already indexed, use “Rebuild graph” above to extract them now."
      />
    )
  }

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <SegmentedControl
          options={[
            ['explore', `Entities (${formatNumber(facets?.total_entities ?? 0)})`],
            ['relations', `Relations (${formatNumber(facets?.total_relations ?? 0)})`],
          ]}
          value={view}
          onChange={setView}
        />
        <span className="text-[11px] text-ink-faint">
          {view === 'explore'
            ? 'Extracted by the LLM during ingestion — this is exactly what GRAPH and MULTIHOP answers traverse.'
            : 'Every extracted relationship, with the passage it came from.'}
        </span>
      </div>

      {view === 'relations' ? (
        <RelationsView
          kbId={kbId}
          facets={facets}
          onOpen={(id) => focus(id)}
          onEvidence={setEvidenceChunk}
        />
      ) : (
        <div className="grid gap-3 lg:grid-cols-[280px_minmax(0,1fr)]">
          <div className="lg:max-h-[540px]">
            <EntityList kbId={kbId} facets={facets} selectedId={selectedId} onOpen={(id) => focus(id)} />
          </div>

          <div className="min-w-0 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex min-w-0 items-center gap-1.5 text-[11px] text-ink-muted">
                <Share2 className="h-3 w-3 shrink-0 text-ink-faint" />
                {centerId ? (
                  <span className="truncate">
                    Around <span className="text-ink">{center?.name || centerId}</span> ·{' '}
                    {hiddenNeighbours > 0
                      ? `${formatNumber(drawnNeighbours)} strongest of ${formatNumber(
                          center.degree
                        )} neighbours`
                      : `${formatNumber(drawnNeighbours)} neighbours`}
                    , {formatNumber(graph.edges.length)} relations
                  </span>
                ) : (
                  <span className="truncate">
                    {graph.nodes.length} most-connected entities, {graph.edges.length} relations
                  </span>
                )}
              </span>
              {graphLoading && <Spinner className="h-3 w-3" />}

              <span className="ml-auto flex items-center gap-2">
                {centerId && (
                  <SegmentedControl
                    options={[
                      [1, '1 hop'],
                      [2, '2 hops'],
                    ]}
                    value={hops}
                    onChange={changeHops}
                  />
                )}
                <SegmentedControl
                  options={[
                    ['force', 'Force'],
                    ['rings', 'Rings'],
                  ]}
                  value={layout}
                  onChange={setLayout}
                />
                {centerId && (
                  <Button variant="ghost" size="sm" onClick={showOverview}>
                    Overview
                  </Button>
                )}
              </span>
            </div>

            <GraphCanvas
              graph={graph}
              centerId={centerId}
              selectedId={selectedId}
              layout={layout}
              onSelect={select}
              onExpand={expand}
              onClear={() => setSelectedId(null)}
            />

            <EntityDetail
              detail={selectedId ? detail : null}
              loading={detailLoading}
              onOpen={(id) => focus(id)}
              onFocus={(id) => focus(id)}
              onExpand={expand}
              onEvidence={setEvidenceChunk}
            />
          </div>
        </div>
      )}

      <ChunkPreview
        chunkId={evidenceChunk}
        term={detail?.entity?.name}
        onClose={() => setEvidenceChunk(null)}
      />
    </div>
  )
}
