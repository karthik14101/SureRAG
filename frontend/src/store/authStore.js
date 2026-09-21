import { create } from 'zustand'
import { authApi } from '../api/auth'
import { getToken, setToken } from '../api/client'

export const useAuthStore = create((set) => ({
  user: null,
  loading: true,
  error: null,

  /** Restore a session from the stored token on first paint. */
  async bootstrap() {
    if (!getToken()) {
      set({ user: null, loading: false })
      return
    }
    try {
      const user = await authApi.me()
      set({ user, loading: false, error: null })
    } catch {
      setToken(null)
      set({ user: null, loading: false })
    }
  },

  async signIn(username, password) {
    set({ error: null })
    const user = await authApi.signIn({ username, password })
    set({ user, error: null })
    return user
  },

  async signUp(username, password, displayName) {
    set({ error: null })
    const user = await authApi.signUp({ username, password, displayName })
    set({ user, error: null })
    return user
  },

  async signOut() {
    await authApi.signOut()
    set({ user: null })
  },

  clear() {
    setToken(null)
    set({ user: null, loading: false })
  },
}))
