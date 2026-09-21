import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Layers, Network, Search, Split, Square, Zap } from 'lucide-react'
import { cn } from '../../lib/cn'

const ROUTE_OPTIONS = [
  { value: null, label: 'Auto', icon: Zap, hint: 'Let the router choose (recommended)' },
  { value: 'VECTOR', label: 'Vector', icon: Search, hint: 'Force semantic passage search' },
  { value: 'GRAPH', label: 'Graph', icon: Network, hint: 'Force knowledge graph traversal' },
  { value: 'HYBRID', label: 'Hybrid', icon: Layers, hint: 'Force both together' },
  { value: 'MULTIHOP', label: 'Multi-hop', icon: Split, hint: 'Force sub-question decomposition' },
]

const MAX_HEIGHT = 200

export function Composer({ onSend, onStop, sending, disabled, placeholder }) {
  const [value, setValue] = useState('')
  const [forceRoute, setForceRoute] = useState(null)
  const [showRoutes, setShowRoutes] = useState(false)
  const textareaRef = useRef(null)

  // Grow the textarea with its content, up to a cap.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(MAX_HEIGHT, el.scrollHeight)}px`
  }, [value])

  function submit() {
    const question = value.trim()
    if (!question || sending || disabled) return
    onSend(question, forceRoute)
    setValue('')
  }

  function onKeyDown(event) {
    // Enter sends; Shift+Enter inserts a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const activeRoute = ROUTE_OPTIONS.find((option) => option.value === forceRoute)
  const ActiveIcon = activeRoute?.icon || Zap

  return (
    <div className="border-t border-line bg-canvas/85 px-4 py-3 backdrop-blur">
      <div className="mx-auto max-w-4xl">
        <div
          className={cn(
            'relative rounded-xl2 border bg-surface-2 transition-colors',
            disabled ? 'border-line opacity-60' : 'border-line focus-within:border-brand/50'
          )}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={disabled}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder || 'Ask anything about these documents...'}
            className={cn(
              'w-full resize-none bg-transparent px-4 pt-3.5 pb-12 text-sm text-ink',
              'placeholder:text-ink-faint focus:outline-none'
            )}
          />

          <div className="absolute inset-x-2 bottom-2 flex items-center justify-between gap-2">
            {/* Route override: a debugging affordance, deliberately understated. */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowRoutes((open) => !open)}
                disabled={disabled}
                title={activeRoute?.hint}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] transition-colors',
                  forceRoute
                    ? 'bg-brand/15 text-brand-soft'
                    : 'text-ink-faint hover:bg-surface-3 hover:text-ink-muted'
                )}
              >
                <ActiveIcon className="h-3 w-3" />
                {activeRoute?.label || 'Auto'}
              </button>

              {showRoutes && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowRoutes(false)} />
                  <div className="absolute bottom-full left-0 z-20 mb-1.5 w-52 overflow-hidden rounded-lg border border-line bg-surface shadow-xl">
                    {ROUTE_OPTIONS.map((option) => {
                      const Icon = option.icon
                      return (
                        <button
                          key={option.label}
                          onClick={() => {
                            setForceRoute(option.value)
                            setShowRoutes(false)
                          }}
                          className={cn(
                            'flex w-full items-start gap-2 px-3 py-2 text-left transition-colors hover:bg-surface-2',
                            forceRoute === option.value && 'bg-surface-2'
                          )}
                        >
                          <Icon className="mt-0.5 h-3 w-3 shrink-0 text-ink-muted" />
                          <span className="min-w-0">
                            <span className="block text-[11px] font-medium text-ink">
                              {option.label}
                            </span>
                            <span className="block text-[10px] leading-snug text-ink-faint">
                              {option.hint}
                            </span>
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </>
              )}
            </div>

            {sending ? (
              <button
                onClick={onStop}
                className="inline-flex items-center gap-1.5 rounded-lg bg-surface-3 px-3 py-1.5 text-[11px] font-medium text-ink hover:bg-[#2b3752]"
              >
                <Square className="h-3 w-3 fill-current" />
                Stop
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!value.trim() || disabled}
                className={cn(
                  'inline-flex h-8 w-8 items-center justify-center rounded-lg transition-colors',
                  value.trim() && !disabled
                    ? 'bg-brand text-white hover:bg-brand-dim'
                    : 'bg-surface-3 text-ink-faint'
                )}
                aria-label="Send"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        <p className="mt-1.5 text-center text-[10px] text-ink-faint">
          Enter to send · Shift+Enter for a new line · Answers cite the sources they used
        </p>
      </div>
    </div>
  )
}
