import { create } from 'zustand'

let nextId = 0

/** Minimal toast queue. Toasts auto-dismiss unless they are errors. */
export const useToast = create((set, get) => ({
  toasts: [],

  push(message, tone = 'info', ttl = 4200) {
    const id = ++nextId
    set((state) => ({ toasts: [...state.toasts, { id, message, tone }] }))
    if (tone !== 'error') {
      setTimeout(() => get().dismiss(id), ttl)
    } else {
      setTimeout(() => get().dismiss(id), 9000)
    }
    return id
  },

  success: (message) => get().push(message, 'success'),
  error: (message) => get().push(message, 'error'),
  info: (message) => get().push(message, 'info'),

  dismiss(id) {
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }))
  },
}))
