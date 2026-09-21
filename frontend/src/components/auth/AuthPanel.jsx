import { useState } from 'react'
import { Brain, Network, ShieldCheck, Sparkles } from 'lucide-react'
import { Button, Input } from '../ui'
import { useAuthStore } from '../../store/authStore'
import { cn } from '../../lib/cn'

const FEATURES = [
  {
    icon: Network,
    title: 'Hybrid graph retrieval',
    body: 'Dense vectors, BM25 keywords and a knowledge graph, fused per question.',
  },
  {
    icon: ShieldCheck,
    title: 'Evidence verified',
    body: 'Answers are checked for sufficient evidence before they are written.',
  },
  {
    icon: Sparkles,
    title: 'Runs on your machine',
    body: 'Local embeddings, local databases, your documents never leave.',
  },
]

export function AuthPanel() {
  const [mode, setMode] = useState('signin')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const signIn = useAuthStore((s) => s.signIn)
  const signUp = useAuthStore((s) => s.signUp)

  const isSignUp = mode === 'signup'

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)

    if (!username.trim() || !password) {
      setError('Enter both a username and a password.')
      return
    }
    if (isSignUp && password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }

    setBusy(true)
    try {
      if (isSignUp) await signUp(username.trim(), password, displayName.trim())
      else await signIn(username.trim(), password)
    } catch (err) {
      setError(err.message || 'Something went wrong. Try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-full grid lg:grid-cols-[1.1fr_1fr]">
      {/* Brand panel */}
      <div className="relative hidden lg:flex flex-col justify-between p-12 overflow-hidden border-r border-line">
        <div
          className="absolute inset-0 opacity-70"
          style={{
            background:
              'radial-gradient(900px 500px at 15% 10%, rgba(99,102,241,0.18), transparent 60%),' +
              'radial-gradient(700px 420px at 85% 85%, rgba(34,211,238,0.12), transparent 60%)',
          }}
        />
        <div className="relative">
          <div className="flex items-center gap-2.5">
            <div className="rounded-lg bg-brand/20 border border-brand/30 p-2">
              <Brain className="h-5 w-5 text-brand-soft" />
            </div>
            <span className="text-lg font-semibold tracking-tight">SURE-GraphRAG</span>
          </div>
          <p className="mt-2 text-xs text-ink-faint tracking-wide uppercase">
            Adaptive Hybrid Graph-Agentic Retrieval
          </p>
        </div>

        <div className="relative space-y-7 max-w-md">
          <h1 className="text-3xl font-semibold leading-tight tracking-tight">
            Ask your documents.
            <br />
            <span className="text-ink-muted">Get answers you can verify.</span>
          </h1>
          <div className="space-y-4">
            {FEATURES.map(({ icon: Icon, title, body }) => (
              <div key={title} className="flex gap-3">
                <div className="mt-0.5 rounded-md bg-surface-2 border border-line p-1.5 h-fit">
                  <Icon className="h-3.5 w-3.5 text-brand-soft" />
                </div>
                <div>
                  <p className="text-sm font-medium text-ink">{title}</p>
                  <p className="text-xs text-ink-muted leading-relaxed">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <p className="relative text-xs text-ink-faint">
          PDF, Word, text, JSON, images and ZIP archives.
        </p>
      </div>

      {/* Form panel */}
      <div className="flex items-center justify-center p-6 sm:p-12">
        <div className="w-full max-w-sm space-y-7">
          <div className="lg:hidden flex items-center gap-2.5">
            <div className="rounded-lg bg-brand/20 border border-brand/30 p-2">
              <Brain className="h-5 w-5 text-brand-soft" />
            </div>
            <span className="text-lg font-semibold tracking-tight">SURE-GraphRAG</span>
          </div>

          <div className="space-y-1.5">
            <h2 className="text-xl font-semibold tracking-tight">
              {isSignUp ? 'Create your account' : 'Welcome back'}
            </h2>
            <p className="text-sm text-ink-muted">
              {isSignUp
                ? 'Your knowledge bases and chats stay private to your account.'
                : 'Sign in to reach your knowledge bases.'}
            </p>
          </div>

          {/* Mode switch */}
          <div className="flex rounded-lg bg-surface-2 border border-line p-1">
            {[
              ['signin', 'Sign in'],
              ['signup', 'Sign up'],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setMode(value)
                  setError(null)
                }}
                className={cn(
                  'flex-1 rounded-md py-1.5 text-xs font-medium transition-colors',
                  mode === value
                    ? 'bg-surface-3 text-ink shadow-sm'
                    : 'text-ink-muted hover:text-ink'
                )}
              >
                {label}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Username"
              name="username"
              autoComplete="username"
              placeholder="your-name"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
            />
            {isSignUp && (
              <Input
                label="Display name (optional)"
                name="displayName"
                placeholder="How you want to be greeted"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
              />
            )}
            <Input
              label="Password"
              name="password"
              type="password"
              autoComplete={isSignUp ? 'new-password' : 'current-password'}
              placeholder={isSignUp ? 'At least 8 characters' : '********'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            {error && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2.5">
                <p className="text-xs text-danger leading-relaxed">{error}</p>
              </div>
            )}

            <Button type="submit" size="lg" loading={busy} className="w-full justify-center">
              {isSignUp ? 'Create account' : 'Sign in'}
            </Button>
          </form>

          <p className="text-center text-xs text-ink-faint leading-relaxed">
            Accounts are stored locally in SQLite on this machine.
            <br />
            Passwords are hashed with bcrypt.
          </p>
        </div>
      </div>
    </div>
  )
}
