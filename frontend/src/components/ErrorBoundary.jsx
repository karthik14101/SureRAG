import { Component } from 'react'
import { AlertTriangle, RotateCw } from 'lucide-react'

/**
 * Catches a render error instead of letting it blank the page.
 *
 * Without one of these, a single bad render anywhere unmounts the whole tree
 * and leaves a white screen with the reason only in the console. That is a real
 * risk here: the trace metadata the chat renders comes from a server that adds
 * fields as the engine grows, and an older client meeting a newer shape should
 * degrade to a message, not to nothing.
 *
 * Two deliberate limits, because boundaries are often expected to do more than
 * they can. They catch errors thrown while RENDERING; errors inside event
 * handlers, timers or promise callbacks never reach them. And they need to be
 * class components -- React has no hook equivalent.
 */
export class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // The console keeps the component stack, which is what actually locates it.
    console.error('Render error caught by boundary:', error, info?.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    const compact = this.props.compact

    return (
      <div
        className={
          compact
            ? 'flex h-full items-center justify-center p-6'
            : 'flex min-h-screen items-center justify-center p-6'
        }
      >
        <div className="w-full max-w-md rounded-xl2 border border-danger/30 bg-surface p-5 text-center">
          <div className="mx-auto mb-3 w-fit rounded-lg border border-danger/30 bg-danger/10 p-2">
            <AlertTriangle className="h-5 w-5 text-danger" />
          </div>

          <h2 className="text-sm font-semibold text-ink">This screen stopped working</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
            Something went wrong while drawing the page. Your documents and chats are
            unaffected — nothing was lost.
          </p>

          {/* Shown because this app runs on the user's own machine: the message
              is the fastest route to a fix, and there is no one to leak it to. */}
          <p className="mt-3 break-words rounded-lg bg-surface-2 px-3 py-2 text-left font-mono text-[11px] leading-relaxed text-ink-faint">
            {String(error?.message || error)}
          </p>

          <div className="mt-4 flex items-center justify-center gap-2">
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-xs text-ink transition-colors hover:bg-surface-3"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-90"
            >
              <RotateCw className="h-3 w-3" />
              Reload
            </button>
          </div>
        </div>
      </div>
    )
  }
}
