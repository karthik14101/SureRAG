import { useEffect, useState } from 'react'
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'
import { Activity, Brain, LogOut } from 'lucide-react'
import { useAuthStore } from './store/authStore'
import { healthApi } from './api/health'
import { cn } from './lib/cn'
import { Badge, Button, Modal, Spinner, ToastViewport } from './components/ui'
import { AuthPanel } from './components/auth/AuthPanel'
import { DashboardPage } from './pages/DashboardPage'
import { KBDetailPage } from './pages/KBDetailPage'
import { ChatPage } from './pages/ChatPage'

/** Diagnostics panel — the fastest way to see which service is not running. */
function HealthModal({ open, onClose }) {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open) return
    setLoading(true)
    healthApi
      .deep()
      .then(setHealth)
      .catch((err) => setHealth({ ok: false, components: [], error: err.message }))
      .finally(() => setLoading(false))
  }, [open])

  return (
    <Modal open={open} onClose={onClose} title="System status" width="max-w-lg">
      {loading && <Spinner label="Checking services..." />}

      {health?.error && <p className="text-sm text-danger">{health.error}</p>}

      {health?.components?.length > 0 && (
        <div className="space-y-2">
          {health.components.map((component) => (
            <div
              key={component.name}
              className="flex items-start gap-3 rounded-lg border border-line bg-surface-2 px-3 py-2.5"
            >
              <span
                className={cn(
                  'mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full',
                  component.ok ? 'bg-success' : 'bg-danger'
                )}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium capitalize text-ink">
                    {component.name}
                  </span>
                  {component.latency_ms > 0 && (
                    <span className="text-[10px] tabular-nums text-ink-faint">
                      {component.latency_ms} ms
                    </span>
                  )}
                </div>
                <p className="mt-0.5 break-words text-[11px] leading-relaxed text-ink-muted">
                  {component.detail}
                </p>
              </div>
            </div>
          ))}

          {health.config && (
            <div className="mt-3 rounded-lg border border-line bg-surface-2 px-3 py-2.5">
              <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-ink-faint">
                Active configuration
              </p>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                {Object.entries(health.config).map(([key, value]) => (
                  <div key={key} className="contents">
                    <dt className="truncate text-ink-faint">{key}</dt>
                    <dd className="truncate text-ink-muted">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
      )}
    </Modal>
  )
}

function TopBar() {
  const user = useAuthStore((state) => state.user)
  const signOut = useAuthStore((state) => state.signOut)
  const location = useLocation()
  const [showHealth, setShowHealth] = useState(false)

  // The chat view manages its own full-height layout and needs no chrome above it.
  const compact = location.pathname.startsWith('/chat/')

  return (
    <>
      <header
        className={cn(
          'flex shrink-0 items-center gap-3 border-b border-line bg-surface/70 px-4 backdrop-blur',
          compact ? 'h-11' : 'h-13 py-2.5'
        )}
      >
        <Link to="/" className="flex items-center gap-2">
          <div className="rounded-md bg-brand/20 p-1.5">
            <Brain className="h-4 w-4 text-brand-soft" />
          </div>
          <span className="text-sm font-semibold tracking-tight">SURE-GraphRAG</span>
        </Link>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setShowHealth(true)}
            className="rounded-md p-1.5 text-ink-faint transition-colors hover:bg-surface-2 hover:text-ink"
            title="System status"
          >
            <Activity className="h-4 w-4" />
          </button>

          <Badge>{user?.display_name || user?.username}</Badge>

          <Button variant="ghost" size="icon" onClick={signOut} title="Sign out">
            <LogOut className="h-3.5 w-3.5" />
          </Button>
        </div>
      </header>

      <HealthModal open={showHealth} onClose={() => setShowHealth(false)} />
    </>
  )
}

function Shell() {
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/kb/:kbId" element={<KBDetailPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  )
}

/** Chat gets the full viewport: its own sidebar, scroll area and composer. */
function ChatShell() {
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="min-h-0 flex-1">
        <Routes>
          <Route path="/chat/:kbId" element={<ChatPage />} />
        </Routes>
      </div>
    </div>
  )
}

function Router() {
  const location = useLocation()
  return location.pathname.startsWith('/chat/') ? <ChatShell /> : <Shell />
}

export default function App() {
  const { user, loading, bootstrap, clear } = useAuthStore()

  useEffect(() => {
    bootstrap()
    // The API client raises this when any request comes back 401.
    const onUnauthorized = () => clear()
    window.addEventListener('sure:unauthorized', onUnauthorized)
    return () => window.removeEventListener('sure:unauthorized', onUnauthorized)
  }, [bootstrap, clear])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner label="Starting SURE-GraphRAG..." />
      </div>
    )
  }

  return (
    <>
      {user ? (
        <BrowserRouter>
          <Router />
        </BrowserRouter>
      ) : (
        <AuthPanel />
      )}
      <ToastViewport />
    </>
  )
}
