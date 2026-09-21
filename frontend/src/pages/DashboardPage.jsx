import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Database,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  Network,
  Plus,
  Trash2,
} from 'lucide-react'
import { useKbStore } from '../store/kbStore'
import { useToast } from '../hooks/useToast'
import { formatNumber, formatRelativeTime } from '../lib/format'
import { Button, ConfirmDialog, EmptyState, Input, Modal, Spinner, Textarea } from '../components/ui'

function StatPill({ icon: Icon, value, label }) {
  return (
    <div className="flex items-center gap-1.5" title={label}>
      <Icon className="h-3 w-3 text-ink-faint" />
      <span className="text-[11px] tabular-nums text-ink-muted">{formatNumber(value)}</span>
    </div>
  )
}

function KBCard({ kb, onOpen, onChat, onDelete }) {
  return (
    <div className="group flex flex-col rounded-xl2 border border-line bg-surface p-4 transition-colors hover:border-brand/35">
      <div className="flex items-start justify-between gap-3">
        <button onClick={() => onOpen(kb.id)} className="min-w-0 flex-1 text-left">
          <h3 className="truncate text-sm font-semibold text-ink group-hover:text-brand-soft">
            {kb.name}
          </h3>
          <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-ink-muted">
            {kb.description || 'No description'}
          </p>
        </button>
        <button
          onClick={() => onDelete(kb)}
          className="shrink-0 rounded p-1 text-ink-faint opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger group-hover:opacity-100"
          aria-label={`Delete ${kb.name}`}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="mt-3.5 flex flex-wrap items-center gap-3">
        <StatPill icon={FileText} value={kb.doc_count} label="Documents" />
        <StatPill icon={Database} value={kb.chunk_count} label="Indexed chunks" />
        <StatPill icon={ImageIcon} value={kb.image_count} label="Extracted images" />
        <StatPill icon={Network} value={kb.entity_count} label="Graph entities" />
      </div>

      <div className="mt-4 flex items-center gap-2 border-t border-line pt-3">
        <Button size="sm" variant="secondary" onClick={() => onOpen(kb.id)} className="flex-1 justify-center">
          Manage
        </Button>
        <Button
          size="sm"
          onClick={() => onChat(kb.id)}
          disabled={kb.doc_count === 0}
          title={kb.doc_count === 0 ? 'Upload a document first' : 'Open chat'}
          className="flex-1 justify-center"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Chat
        </Button>
      </div>

      <p className="mt-2 text-[10px] text-ink-faint">
        Updated {formatRelativeTime(kb.updated_at)}
      </p>
    </div>
  )
}

export function DashboardPage() {
  const navigate = useNavigate()
  const { kbs, loading, load, create, remove } = useKbStore()
  const toast = useToast()

  const [showCreate, setShowCreate] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [creating, setCreating] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    load()
  }, [load])

  async function handleCreate(event) {
    event.preventDefault()
    if (!name.trim()) return
    setCreating(true)
    try {
      const kb = await create(name.trim(), description.trim())
      toast.success(`Created "${kb.name}".`)
      setShowCreate(false)
      setName('')
      setDescription('')
      navigate(`/kb/${kb.id}`)
    } catch (err) {
      toast.error(err.message || 'Could not create the knowledge base.')
    } finally {
      setCreating(false)
    }
  }

  async function confirmDelete() {
    setDeleting(true)
    try {
      await remove(pendingDelete.id)
      toast.success(`Deleted "${pendingDelete.name}".`)
      setPendingDelete(null)
    } catch (err) {
      toast.error(err.message || 'Could not delete the knowledge base.')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-5 py-8">
      <header className="mb-7 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Knowledge bases</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Each one is an isolated collection of documents you can chat with.
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" />
          New knowledge base
        </Button>
      </header>

      {loading && !kbs.length && (
        <div className="py-12">
          <Spinner label="Loading..." />
        </div>
      )}

      {!loading && !kbs.length && (
        <div className="rounded-xl2 border border-dashed border-line bg-surface/50">
          <EmptyState
            icon={Database}
            title="No knowledge bases yet"
            description="Create one, upload your documents, and start asking questions with verifiable citations."
            action={
              <Button onClick={() => setShowCreate(true)}>
                <Plus className="h-4 w-4" />
                Create your first one
              </Button>
            }
          />
        </div>
      )}

      {kbs.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {kbs.map((kb) => (
            <KBCard
              key={kb.id}
              kb={kb}
              onOpen={(id) => navigate(`/kb/${id}`)}
              onChat={(id) => navigate(`/chat/${id}`)}
              onDelete={setPendingDelete}
            />
          ))}
        </div>
      )}

      <Modal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="New knowledge base"
        description="Group related documents so questions stay focused."
        footer={
          <>
            <Button variant="ghost" size="sm" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleCreate} loading={creating} disabled={!name.trim()}>
              Create
            </Button>
          </>
        }
      >
        <form onSubmit={handleCreate} className="space-y-4">
          <Input
            label="Name"
            name="name"
            autoFocus
            placeholder="Product documentation"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <Textarea
            label="Description (optional)"
            name="description"
            rows={3}
            placeholder="What lives in here?"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </form>
      </Modal>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        onClose={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
        loading={deleting}
        title="Delete this knowledge base?"
        message={`"${pendingDelete?.name}", its ${pendingDelete?.doc_count ?? 0} document(s), every indexed chunk, the graph entities and all chats in it will be permanently deleted. This cannot be undone.`}
      />
    </div>
  )
}
