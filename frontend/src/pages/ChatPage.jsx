import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Brain, MessageSquare, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { kbApi } from '../api/kb'
import { useChatStore } from '../store/chatStore'
import { useToast } from '../hooks/useToast'
import { cn } from '../lib/cn'
import { Button, EmptyState, Spinner } from '../components/ui'
import { SessionSidebar } from '../components/chat/SessionSidebar'
import { MessageBubble } from '../components/chat/MessageBubble'
import { Composer } from '../components/chat/Composer'
import { CitationDrawer } from '../components/citations/CitationChip'

const SUGGESTIONS = [
  'Summarise the key points across these documents.',
  'What are the main entities mentioned, and how do they relate?',
  'List every requirement or constraint stated.',
]

export function ChatPage() {
  const { kbId } = useParams()
  const navigate = useNavigate()
  const toast = useToast()

  const {
    sessions,
    activeSessionId,
    messages,
    loadingSessions,
    loadingMessages,
    sending,
    streaming,
    error,
    loadSessions,
    selectSession,
    createSession,
    renameSession,
    deleteSession,
    deleteMessage,
    send,
    stop,
    clearError,
    reset,
  } = useChatStore()

  const [kb, setKb] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [openCitation, setOpenCitation] = useState(null)
  const scrollRef = useRef(null)
  const bottomRef = useRef(null)
  const pinnedToBottom = useRef(true)

  useEffect(() => {
    kbApi
      .get(kbId)
      .then(setKb)
      .catch(() => {
        toast.error('That knowledge base is not available.')
        navigate('/')
      })
    loadSessions(kbId)
    return () => reset()
  }, [kbId, loadSessions, navigate, reset, toast])

  useEffect(() => {
    if (error) {
      toast.error(error)
      clearError()
    }
  }, [error, toast, clearError])

  // Auto-scroll while streaming, but stop fighting the user if they scroll up.
  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight
    pinnedToBottom.current = distance < 120
  }, [])

  useEffect(() => {
    if (pinnedToBottom.current) {
      bottomRef.current?.scrollIntoView({ behavior: streaming ? 'auto' : 'smooth' })
    }
  }, [messages, streaming?.text, streaming])

  async function startChat(question) {
    try {
      await createSession(kbId)
      // Zustand has applied the new session id synchronously by now.
      if (question) await useChatStore.getState().send(question)
    } catch (err) {
      toast.error(err.message || 'Could not start a chat.')
    }
  }

  async function handleSend(question, forceRoute) {
    if (!activeSessionId) {
      await createSession(kbId)
    }
    await send(question, forceRoute)
  }

  async function handleDeleteMessage(messageId) {
    try {
      await deleteMessage(messageId)
    } catch (err) {
      toast.error(err.message || 'Could not delete that message.')
    }
  }

  const hasThread = Boolean(activeSessionId)
  const showSuggestions = hasThread && !messages.length && !sending

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <aside
        className={cn(
          'flex shrink-0 flex-col border-r border-line bg-surface/60 transition-[width] duration-200',
          sidebarOpen ? 'w-64' : 'w-0 overflow-hidden'
        )}
      >
        <div className="border-b border-line px-3 py-3">
          <button
            onClick={() => navigate(`/kb/${kbId}`)}
            className="inline-flex items-center gap-1.5 text-xs text-ink-muted transition-colors hover:text-ink"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span className="truncate">{kb?.name || 'Knowledge base'}</span>
          </button>
        </div>

        <SessionSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          loading={loadingSessions}
          onSelect={selectSession}
          onCreate={() => startChat()}
          onRename={renameSession}
          onDelete={deleteSession}
        />
      </aside>

      {/* Main column */}
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-line px-4 py-2.5">
          <button
            onClick={() => setSidebarOpen((open) => !open)}
            className="rounded-md p-1.5 text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
            aria-label={sidebarOpen ? 'Hide chats' : 'Show chats'}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeftOpen className="h-4 w-4" />
            )}
          </button>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-medium text-ink">
              {sessions.find((s) => s.id === activeSessionId)?.title || 'New chat'}
            </h1>
            <p className="truncate text-[10px] text-ink-faint">
              {kb?.name} · {kb?.doc_count ?? 0} document{kb?.doc_count === 1 ? '' : 's'} ·{' '}
              {kb?.chunk_count ?? 0} chunks
            </p>
          </div>
        </header>

        <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-4xl space-y-6 px-4 py-6">
            {!hasThread && !loadingSessions && (
              <div className="pt-10">
                <EmptyState
                  icon={Brain}
                  title={`Ask about ${kb?.name || 'your documents'}`}
                  description="Every answer cites the passages it came from, and shows how the evidence was gathered."
                  action={
                    <Button onClick={() => startChat()}>
                      <MessageSquare className="h-4 w-4" />
                      Start a chat
                    </Button>
                  }
                />
                <div className="mx-auto mt-2 grid max-w-lg gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => startChat(suggestion)}
                      className="rounded-lg border border-line bg-surface px-3.5 py-2.5 text-left text-xs text-ink-muted transition-colors hover:border-brand/40 hover:text-ink"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {loadingMessages && <Spinner label="Loading messages..." />}

            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onOpenCitation={setOpenCitation}
                // A turn that has not been saved yet has no id the server
                // would recognise, and nothing is removable mid-answer.
                deletable={
                  message.role === 'user' &&
                  !sending &&
                  !String(message.id).startsWith('pending-')
                }
                onDelete={handleDeleteMessage}
              />
            ))}

            {streaming && (
              <MessageBubble
                message={{
                  id: 'streaming',
                  role: 'assistant',
                  content: streaming.text,
                  citations: [],
                  media: [],
                }}
                streaming
                liveTraces={streaming.traces}
                liveStatus={streaming.status}
                onOpenCitation={setOpenCitation}
              />
            )}

            {showSuggestions && (
              <div className="grid gap-2 pt-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSend(suggestion)}
                    className="rounded-lg border border-line bg-surface px-3.5 py-2.5 text-left text-xs text-ink-muted transition-colors hover:border-brand/40 hover:text-ink"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        <Composer
          onSend={handleSend}
          onStop={stop}
          sending={sending}
          disabled={!kb}
          placeholder={`Ask about ${kb?.name || 'these documents'}...`}
        />
      </main>

      <CitationDrawer citation={openCitation} onClose={() => setOpenCitation(null)} />
    </div>
  )
}
