import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  Database,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  Network,
  RefreshCw,
  Share2,
} from 'lucide-react'
import { kbApi } from '../api/kb'
import { useIngestJobs } from '../hooks/useIngestJobs'
import { useToast } from '../hooks/useToast'
import { useKbStore } from '../store/kbStore'
import { formatBytes, formatNumber } from '../lib/format'
import { Badge, Button, Spinner } from '../components/ui'
import { Dropzone, UploadQueue } from '../components/kb/Dropzone'
import { DocumentTable } from '../components/kb/DocumentTable'
import { ChunkExplorer } from '../components/kb/ChunkExplorer'
import { ImageGallery } from '../components/media/ImageGallery'

function StatCard({ icon: Icon, label, value, hint }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3.5">
      <div className="flex items-center gap-1.5 text-ink-faint">
        <Icon className="h-3 w-3" />
        <span className="text-[10px] font-medium uppercase tracking-wide">{label}</span>
      </div>
      <p className="mt-1.5 text-lg font-semibold tabular-nums text-ink">{value}</p>
      {hint && <p className="mt-0.5 text-[10px] text-ink-faint">{hint}</p>}
    </div>
  )
}

export function KBDetailPage() {
  const { kbId } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  const refreshOne = useKbStore((state) => state.refreshOne)

  const [kb, setKb] = useState(null)
  const [stats, setStats] = useState(null)
  const [documents, setDocuments] = useState([])
  const [media, setMedia] = useState([])
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('documents')
  const [rebuilding, setRebuilding] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [kbData, docs, statsData] = await Promise.all([
        kbApi.get(kbId),
        kbApi.documents(kbId),
        kbApi.stats(kbId).catch(() => null),
      ])
      setKb(kbData)
      setDocuments(docs)
      setStats(statsData)
      refreshOne(kbId)
    } catch (err) {
      toast.error(err.message || 'Could not load this knowledge base.')
      navigate('/')
    } finally {
      setLoading(false)
    }
  }, [kbId, navigate, refreshOne, toast])

  const { jobs, isIngesting, track, dismiss } = useIngestJobs(kbId, { onComplete: refresh })

  useEffect(() => {
    setLoading(true)
    refresh()
  }, [refresh])

  // Keep the table live while files are being processed.
  useEffect(() => {
    if (!isIngesting) return undefined
    const timer = setInterval(() => {
      kbApi.documents(kbId).then(setDocuments).catch(() => {})
    }, 2000)
    return () => clearInterval(timer)
  }, [isIngesting, kbId])

  useEffect(() => {
    if (tab !== 'images') return
    kbApi.media(kbId).then(setMedia).catch(() => {})
  }, [tab, kbId])

  async function rebuildGraph() {
    setRebuilding(true)
    try {
      const result = await kbApi.rebuildGraph(kbId)
      toast.success(
        `Graph rebuilt: ${formatNumber(result.entities)} entities, ${formatNumber(result.relations)} relationships.`
      )
      refresh()
    } catch (err) {
      toast.error(err.message || 'Could not rebuild the graph.')
    } finally {
      setRebuilding(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner label="Loading knowledge base..." />
      </div>
    )
  }

  if (!kb) return null

  return (
    <div className="mx-auto max-w-6xl px-5 py-6">
      <button
        onClick={() => navigate('/')}
        className="mb-4 inline-flex items-center gap-1.5 text-xs text-ink-muted transition-colors hover:text-ink"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        All knowledge bases
      </button>

      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight">{kb.name}</h1>
          {kb.description && (
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-ink-muted">
              {kb.description}
            </p>
          )}
          {stats && !stats.graph_available && (
            <Badge tone="warn" className="mt-2">
              Knowledge graph offline — answers use vector search only
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" onClick={refresh}>
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
          <Button
            size="sm"
            onClick={() => navigate(`/chat/${kbId}`)}
            disabled={!documents.some((d) => d.status === 'ready')}
            title={
              documents.some((d) => d.status === 'ready')
                ? 'Open chat'
                : 'Upload and process a document first'
            }
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Chat
          </Button>
        </div>
      </header>

      {stats && (
        <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-5">
          <StatCard
            icon={FileText}
            label="Documents"
            value={formatNumber(stats.ready_docs)}
            hint={
              stats.pending_docs || stats.failed_docs
                ? `${stats.pending_docs} processing · ${stats.failed_docs} failed`
                : formatBytes(stats.total_bytes)
            }
          />
          <StatCard icon={Database} label="Chunks" value={formatNumber(stats.chunk_count)} hint="Indexed passages" />
          <StatCard icon={ImageIcon} label="Images" value={formatNumber(stats.image_count)} hint="Figures extracted" />
          <StatCard icon={Network} label="Entities" value={formatNumber(stats.entity_count)} hint="In the graph" />
          <StatCard icon={Share2} label="Relations" value={formatNumber(stats.relation_count)} hint="Between entities" />
        </div>
      )}

      <section className="mb-6 space-y-3">
        <Dropzone kbId={kbId} onUploaded={track} />
        <UploadQueue jobs={jobs} onDismiss={dismiss} />
      </section>

      <section className="rounded-xl2 border border-line bg-surface">
        <div className="flex items-center gap-1 border-b border-line px-3 py-2">
          {[
            ['documents', `Documents (${documents.length})`],
            ['chunks', `Chunks (${formatNumber(stats?.chunk_count ?? 0)})`],
            ['images', `Images (${stats?.image_count ?? 0})`],
          ].map(([value, label]) => (
            <button
              key={value}
              onClick={() => setTab(value)}
              className={
                tab === value
                  ? 'rounded-md bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink'
                  : 'rounded-md px-3 py-1.5 text-xs text-ink-muted hover:text-ink'
              }
            >
              {label}
            </button>
          ))}

          {stats?.graph_available && stats.chunk_count > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto"
              onClick={rebuildGraph}
              loading={rebuilding}
              title="Re-extract entities and relationships from the indexed chunks"
            >
              <Network className="h-3.5 w-3.5" />
              Rebuild graph
            </Button>
          )}
        </div>

        {tab === 'documents' && <DocumentTable documents={documents} onDeleted={refresh} />}

        {tab === 'chunks' && <ChunkExplorer kbId={kbId} />}

        {tab === 'images' && (
          <div className="p-4">
            {media.length ? (
              <ImageGallery images={media} compact />
            ) : (
              <p className="py-8 text-center text-sm text-ink-muted">
                No images have been extracted from these documents yet.
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  )
}
