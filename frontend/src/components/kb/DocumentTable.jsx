import { useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  FileArchive,
  Image as ImageIcon,
  Loader2,
  Trash2,
} from 'lucide-react'
import { kbApi } from '../../api/kb'
import { cn } from '../../lib/cn'
import { formatBytes, formatNumber, formatRelativeTime } from '../../lib/format'
import { useToast } from '../../hooks/useToast'
import { Badge, ConfirmDialog, EmptyState, Spinner } from '../ui'
import { FileIcon } from './Dropzone'

const STATUS_META = {
  ready: { label: 'Ready', tone: 'success', icon: CheckCircle2 },
  failed: { label: 'Failed', tone: 'danger', icon: AlertCircle },
  pending: { label: 'Queued', tone: 'neutral', icon: Loader2, spin: true },
  parsing: { label: 'Parsing', tone: 'brand', icon: Loader2, spin: true },
  embedding: { label: 'Embedding', tone: 'brand', icon: Loader2, spin: true },
  graphing: { label: 'Graphing', tone: 'accent', icon: Loader2, spin: true },
}

function StatusBadge({ status }) {
  const meta = STATUS_META[status] || STATUS_META.pending
  const Icon = meta.icon
  return (
    <Badge tone={meta.tone}>
      <Icon className={cn('h-2.5 w-2.5', meta.spin && 'animate-spin')} />
      {meta.label}
    </Badge>
  )
}

export function DocumentTable({ documents, loading, onDeleted }) {
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const toast = useToast()

  async function confirmDelete() {
    setDeleting(true)
    try {
      await kbApi.deleteDocument(pendingDelete.id)
      toast.success(`Removed "${pendingDelete.filename}".`)
      setPendingDelete(null)
      onDeleted?.()
    } catch (err) {
      toast.error(err.message || 'Could not delete the document.')
    } finally {
      setDeleting(false)
    }
  }

  if (loading) {
    return (
      <div className="px-4 py-8">
        <Spinner label="Loading documents..." />
      </div>
    )
  }

  if (!documents?.length) {
    return (
      <EmptyState
        icon={FileArchive}
        title="No documents yet"
        description="Upload files above to start building this knowledge base."
      />
    )
  }

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left">
          <thead>
            <tr className="border-b border-line text-[10px] uppercase tracking-wide text-ink-faint">
              <th className="px-3 py-2 font-medium">Document</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 text-right font-medium">Chunks</th>
              <th className="px-3 py-2 text-right font-medium">Images</th>
              <th className="px-3 py-2 text-right font-medium">Size</th>
              <th className="px-3 py-2 font-medium">Added</th>
              <th className="w-10 px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr
                key={doc.id}
                className="group border-b border-line-soft transition-colors last:border-0 hover:bg-surface-2/60"
              >
                <td className="max-w-xs px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <FileIcon name={doc.filename} className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium text-ink" title={doc.filename}>
                        {doc.filename}
                      </p>
                      {doc.source_path && doc.source_path !== doc.filename && (
                        <p
                          className="truncate text-[10px] text-ink-faint"
                          title={doc.source_path}
                        >
                          from archive: {doc.source_path}
                        </p>
                      )}
                      {doc.error && (
                        <p className="mt-0.5 text-[10px] leading-snug text-danger">{doc.error}</p>
                      )}
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <StatusBadge status={doc.status} />
                </td>
                <td className="px-3 py-2.5 text-right text-xs tabular-nums text-ink-muted">
                  {formatNumber(doc.chunk_count)}
                </td>
                <td className="px-3 py-2.5 text-right text-xs tabular-nums text-ink-muted">
                  {doc.image_count > 0 ? (
                    <span className="inline-flex items-center gap-1">
                      <ImageIcon className="h-3 w-3 text-ink-faint" />
                      {doc.image_count}
                    </span>
                  ) : (
                    '-'
                  )}
                </td>
                <td className="px-3 py-2.5 text-right text-xs tabular-nums text-ink-muted">
                  {formatBytes(doc.size_bytes)}
                </td>
                <td className="px-3 py-2.5 text-xs text-ink-faint">
                  {formatRelativeTime(doc.created_at)}
                </td>
                <td className="px-3 py-2.5">
                  <button
                    onClick={() => setPendingDelete(doc)}
                    className="rounded p-1 text-ink-faint opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger group-hover:opacity-100"
                    aria-label={`Delete ${doc.filename}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        onClose={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
        loading={deleting}
        title="Delete this document?"
        message={`"${pendingDelete?.filename}" will be removed from the vector index, the knowledge graph and disk. Existing chat answers keep their citations but will no longer link to the full passage.`}
      />
    </>
  )
}
