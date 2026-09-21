import { useState } from 'react'
import { Check, MessageSquarePlus, MoreHorizontal, Pencil, Trash2, X } from 'lucide-react'
import { cn } from '../../lib/cn'
import { formatRelativeTime } from '../../lib/format'
import { Button, ConfirmDialog, EmptyState, Spinner } from '../ui'

function SessionRow({ session, active, onSelect, onRename, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(session.title)

  function commit() {
    const title = draft.trim()
    if (title && title !== session.title) onRename(session.id, title)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1 rounded-lg border border-brand/40 bg-surface-2 px-2 py-1.5">
        <input
          value={draft}
          autoFocus
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') commit()
            if (event.key === 'Escape') {
              setDraft(session.title)
              setEditing(false)
            }
          }}
          className="min-w-0 flex-1 bg-transparent text-xs text-ink focus:outline-none"
        />
        <button onClick={commit} className="p-0.5 text-success hover:opacity-80" aria-label="Save">
          <Check className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => {
            setDraft(session.title)
            setEditing(false)
          }}
          className="p-0.5 text-ink-faint hover:text-ink"
          aria-label="Cancel"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'group relative flex items-center gap-1 rounded-lg px-2 py-1.5 transition-colors',
        active ? 'bg-brand/15 border border-brand/25' : 'border border-transparent hover:bg-surface-2'
      )}
    >
      <button onClick={() => onSelect(session.id)} className="min-w-0 flex-1 text-left">
        <p className={cn('truncate text-xs font-medium', active ? 'text-ink' : 'text-ink-muted')}>
          {session.title}
        </p>
        <p className="mt-0.5 text-[10px] text-ink-faint">
          {session.message_count} message{session.message_count === 1 ? '' : 's'} ·{' '}
          {formatRelativeTime(session.updated_at)}
        </p>
      </button>

      <button
        onClick={() => setMenuOpen((open) => !open)}
        className={cn(
          'shrink-0 rounded p-1 text-ink-faint transition-opacity hover:bg-surface-3 hover:text-ink',
          menuOpen ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
        )}
        aria-label="Chat options"
      >
        <MoreHorizontal className="h-3.5 w-3.5" />
      </button>

      {menuOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
          <div className="absolute right-1 top-full z-20 mt-1 w-32 overflow-hidden rounded-lg border border-line bg-surface shadow-xl">
            <button
              onClick={() => {
                setEditing(true)
                setMenuOpen(false)
              }}
              className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-[11px] text-ink-muted hover:bg-surface-2 hover:text-ink"
            >
              <Pencil className="h-3 w-3" />
              Rename
            </button>
            <button
              onClick={() => {
                onDelete(session)
                setMenuOpen(false)
              }}
              className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-[11px] text-danger hover:bg-danger/10"
            >
              <Trash2 className="h-3 w-3" />
              Delete
            </button>
          </div>
        </>
      )}
    </div>
  )
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  loading,
  onSelect,
  onCreate,
  onRename,
  onDelete,
}) {
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  async function confirmDelete() {
    setDeleting(true)
    try {
      await onDelete(pendingDelete.id)
      setPendingDelete(null)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <>
      <div className="flex h-full flex-col">
        <div className="px-3 pb-2 pt-3">
          <Button size="sm" className="w-full justify-center" onClick={onCreate}>
            <MessageSquarePlus className="h-3.5 w-3.5" />
            New chat
          </Button>
        </div>

        <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
          {loading && (
            <div className="px-2 py-4">
              <Spinner label="Loading chats..." />
            </div>
          )}

          {!loading && !sessions.length && (
            <EmptyState
              title="No chats yet"
              description="Start one to ask about the documents in this knowledge base."
            />
          )}

          {sessions.map((session) => (
            <SessionRow
              key={session.id}
              session={session}
              active={session.id === activeSessionId}
              onSelect={onSelect}
              onRename={onRename}
              onDelete={setPendingDelete}
            />
          ))}
        </div>
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        onClose={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
        loading={deleting}
        title="Delete this chat?"
        message={`"${pendingDelete?.title}" and all of its messages will be permanently removed. Your documents are not affected.`}
      />
    </>
  )
}
