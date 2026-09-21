import { create } from 'zustand'
import { kbApi } from '../api/kb'

export const useKbStore = create((set, get) => ({
  kbs: [],
  loading: false,
  error: null,
  activeKbId: null,

  async load() {
    set({ loading: true, error: null })
    try {
      const kbs = await kbApi.list()
      set({ kbs, loading: false })
      // Keep the active selection valid after a deletion elsewhere.
      const { activeKbId } = get()
      if (activeKbId && !kbs.some((kb) => kb.id === activeKbId)) {
        set({ activeKbId: kbs[0]?.id ?? null })
      }
    } catch (err) {
      set({ error: err.message, loading: false })
    }
  },

  setActive(kbId) {
    set({ activeKbId: kbId })
  },

  async create(name, description) {
    const kb = await kbApi.create(name, description)
    set((state) => ({ kbs: [kb, ...state.kbs], activeKbId: kb.id }))
    return kb
  },

  async rename(id, name, description) {
    const updated = await kbApi.rename(id, name, description)
    set((state) => ({ kbs: state.kbs.map((kb) => (kb.id === id ? updated : kb)) }))
    return updated
  },

  async remove(id) {
    await kbApi.remove(id)
    set((state) => {
      const kbs = state.kbs.filter((kb) => kb.id !== id)
      return { kbs, activeKbId: state.activeKbId === id ? (kbs[0]?.id ?? null) : state.activeKbId }
    })
  },

  /** Refresh one knowledge base's counters after an ingestion finishes. */
  async refreshOne(id) {
    try {
      const kb = await kbApi.get(id)
      set((state) => ({ kbs: state.kbs.map((item) => (item.id === id ? kb : item)) }))
    } catch {
      /* the KB may have just been deleted */
    }
  },
}))
