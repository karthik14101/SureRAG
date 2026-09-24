/**
 * Shared UI primitives.
 *
 * Kept in one file on purpose: they are small, always imported together, and
 * splitting them into a dozen files would add friction without adding clarity.
 */
import { useEffect, useRef } from 'react'
import { AlertCircle, CheckCircle2, Info, Loader2, X } from 'lucide-react'
import { cn } from '../../lib/cn'
import { useToast } from '../../hooks/useToast'

/* -------------------------------------------------------------------------- */
/* Button                                                                      */
/* -------------------------------------------------------------------------- */
const BUTTON_VARIANTS = {
  primary:
    'bg-brand text-white hover:bg-brand-dim disabled:hover:bg-brand shadow-sm shadow-brand/25',
  secondary: 'bg-surface-3 text-ink hover:bg-surface-hover border border-line',
  ghost: 'text-ink-muted hover:text-ink hover:bg-surface-2',
  danger: 'bg-danger/15 text-danger border border-danger/30 hover:bg-danger/25',
  outline: 'border border-line text-ink hover:bg-surface-2',
}

const BUTTON_SIZES = {
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-10 px-4 text-sm gap-2',
  lg: 'h-11 px-5 text-sm gap-2',
  icon: 'h-8 w-8 justify-center',
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  className,
  children,
  disabled,
  ...props
}) {
  return (
    <button
      className={cn(
        'inline-flex items-center rounded-lg font-medium transition-colors duration-150',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  )
}

/* -------------------------------------------------------------------------- */
/* Input / Textarea                                                            */
/* -------------------------------------------------------------------------- */
export function Input({ label, error, className, id, ...props }) {
  const inputId = id || props.name
  return (
    <div className="space-y-1.5">
      {label && (
        <label htmlFor={inputId} className="block text-xs font-medium text-ink-muted">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={cn(
          'w-full rounded-lg bg-surface-2 border px-3 py-2.5 text-sm text-ink',
          'placeholder:text-ink-faint transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-brand/50 focus:border-brand',
          error ? 'border-danger/60' : 'border-line',
          className
        )}
        {...props}
      />
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  )
}

export function Textarea({ label, error, className, id, ...props }) {
  const inputId = id || props.name
  return (
    <div className="space-y-1.5">
      {label && (
        <label htmlFor={inputId} className="block text-xs font-medium text-ink-muted">
          {label}
        </label>
      )}
      <textarea
        id={inputId}
        className={cn(
          'w-full rounded-lg bg-surface-2 border px-3 py-2.5 text-sm text-ink resize-none',
          'placeholder:text-ink-faint focus:outline-none focus:ring-2 focus:ring-brand/50',
          error ? 'border-danger/60' : 'border-line',
          className
        )}
        {...props}
      />
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Badge                                                                       */
/* -------------------------------------------------------------------------- */
const BADGE_TONES = {
  neutral: 'bg-surface-3 text-ink-muted border-line',
  brand: 'bg-brand/15 text-brand-soft border-brand/30',
  success: 'bg-success/15 text-success border-success/30',
  warn: 'bg-warn/15 text-warn border-warn/30',
  danger: 'bg-danger/15 text-danger border-danger/30',
  accent: 'bg-accent/15 text-accent border-accent/30',
}

export function Badge({ tone = 'neutral', className, children, ...props }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5',
        'text-[11px] font-medium leading-none whitespace-nowrap',
        BADGE_TONES[tone],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}

/* -------------------------------------------------------------------------- */
/* Spinner / EmptyState                                                        */
/* -------------------------------------------------------------------------- */
export function Spinner({ className, label }) {
  return (
    <div className="flex items-center gap-2 text-ink-muted">
      <Loader2 className={cn('h-4 w-4 animate-spin', className)} />
      {label && <span className="text-sm">{label}</span>}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-14 px-6 text-center">
      {Icon && (
        <div className="rounded-xl bg-surface-2 border border-line p-3">
          <Icon className="h-6 w-6 text-ink-faint" />
        </div>
      )}
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        {description && (
          <p className="text-sm text-ink-muted max-w-sm mx-auto leading-relaxed">
            {description}
          </p>
        )}
      </div>
      {action}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Modal                                                                       */
/* -------------------------------------------------------------------------- */
export function Modal({ open, onClose, title, description, children, footer, width = 'max-w-md' }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') onClose?.()
    }
    document.addEventListener('keydown', onKey)
    // Prevent the page behind the modal from scrolling.
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = previous
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm animate-fade-in"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose?.()
      }}
    >
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        className={cn(
          'w-full rounded-xl2 border border-line bg-surface shadow-2xl animate-slide-up',
          width
        )}
      >
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold text-ink">{title}</h2>
            {description && <p className="text-xs text-ink-muted">{description}</p>}
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-ink-faint hover:text-ink hover:bg-surface-2"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-line px-5 py-3.5">{footer}</div>
        )}
      </div>
    </div>
  )
}

export function ConfirmDialog({ open, onClose, onConfirm, title, message, confirmLabel = 'Delete', loading }) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <>
          <Button variant="ghost" size="sm" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button variant="danger" size="sm" onClick={onConfirm} loading={loading}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <p className="text-sm text-ink-muted leading-relaxed">{message}</p>
    </Modal>
  )
}

/* -------------------------------------------------------------------------- */
/* Toasts                                                                      */
/* -------------------------------------------------------------------------- */
const TOAST_STYLES = {
  success: { icon: CheckCircle2, className: 'border-success/40 bg-success/10 text-success' },
  error: { icon: AlertCircle, className: 'border-danger/40 bg-danger/10 text-danger' },
  info: { icon: Info, className: 'border-brand/40 bg-brand/10 text-brand-soft' },
}

export function ToastViewport() {
  const toasts = useToast((state) => state.toasts)
  const dismiss = useToast((state) => state.dismiss)

  if (!toasts.length) return null

  return (
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 w-[min(24rem,calc(100vw-2rem))]">
      {toasts.map((toast) => {
        const { icon: Icon, className } = TOAST_STYLES[toast.tone] || TOAST_STYLES.info
        return (
          <div
            key={toast.id}
            className={cn(
              'flex items-start gap-2.5 rounded-lg border px-3.5 py-3 backdrop-blur-sm',
              'shadow-lg animate-slide-up',
              className
            )}
          >
            <Icon className="h-4 w-4 shrink-0 mt-0.5" />
            <p className="flex-1 text-xs leading-relaxed text-ink">{toast.message}</p>
            <button
              onClick={() => dismiss(toast.id)}
              className="shrink-0 text-ink-faint hover:text-ink"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )
      })}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Progress                                                                    */
/* -------------------------------------------------------------------------- */
export function ProgressBar({ value = 0, tone = 'brand', className }) {
  const tones = {
    brand: 'bg-brand',
    success: 'bg-success',
    warn: 'bg-warn',
    danger: 'bg-danger',
  }
  return (
    <div className={cn('h-1.5 w-full overflow-hidden rounded-full bg-surface-3', className)}>
      <div
        className={cn('h-full rounded-full transition-[width] duration-300', tones[tone])}
        style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
      />
    </div>
  )
}
