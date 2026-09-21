import { create } from 'zustand'
import { chatApi } from '../api/chat'

/**
 * Chat state.
 *
 * The streaming answer lives in `streaming` (a transient object) rather than in
 * `messages`, so React only re-renders the one growing bubble while tokens
 * arrive instead of the whole thread on every chunk.
 */
export const useChatStore = create((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  loadingSessions: false,
  loadingMessages: false,
  sending: false,
  error: null,

  // Transient state for the in-flight answer.
  streaming: null, // { text, traces, status }
  abortController: null,

  async loadSessions(kbId) {
    set({ loadingSessions: true, error: null })
    try {
      const sessions = await chatApi.listSessions(kbId)
      set({ loadingSessions: false, sessions })
      const { activeSessionId } = get()
      if (activeSessionId && !sessions.some((s) => s.id === activeSessionId)) {
        set({ activeSessionId: null, messages: [] })
      }
    } catch (err) {
      set({ error: err.message, loadingSessions: false })
    }
  },

  async selectSession(sessionId) {
    if (!sessionId) {
      set({ activeSessionId: null, messages: [] })
      return
    }
    set({ activeSessionId: sessionId, loadingMessages: true, messages: [], error: null })
    try {
      const messages = await chatApi.messages(sessionId)
      // Guard against a slow response landing after the user switched threads.
      if (get().activeSessionId === sessionId) {
        set({ messages, loadingMessages: false })
      }
    } catch (err) {
      set({ error: err.message, loadingMessages: false })
    }
  },

  async createSession(kbId, title) {
    const session = await chatApi.createSession(kbId, title)
    set((state) => ({
      sessions: [session, ...state.sessions],
      activeSessionId: session.id,
      messages: [],
    }))
    return session
  },

  async renameSession(id, title) {
    const updated = await chatApi.renameSession(id, title)
    set((state) => ({
      sessions: state.sessions.map((s) => (s.id === id ? updated : s)),
    }))
  },

  async deleteSession(id) {
    await chatApi.deleteSession(id)
    set((state) => {
      const sessions = state.sessions.filter((s) => s.id !== id)
      const wasActive = state.activeSessionId === id
      return {
        sessions,
        activeSessionId: wasActive ? null : state.activeSessionId,
        messages: wasActive ? [] : state.messages,
      }
    })
  },

  /**
   * Send a question and consume the answer stream.
   * Events arrive in order: user_message, trace*, status, token*, done, saved, end.
   */
  async send(question, forceRoute) {
    const sessionId = get().activeSessionId
    if (!sessionId || get().sending) return

    const controller = new AbortController()
    const optimisticId = `pending-${Date.now()}`

    set((state) => ({
      sending: true,
      error: null,
      abortController: controller,
      streaming: { text: '', traces: [], status: null },
      messages: [
        ...state.messages,
        {
          id: optimisticId,
          session_id: sessionId,
          role: 'user',
          content: question,
          created_at: new Date().toISOString(),
          citations: [],
          media: [],
          trace: [],
        },
      ],
    }))

    try {
      await chatApi.askStream(
        sessionId,
        question,
        (event, data) => {
          switch (event) {
            case 'user_message':
              // Swap the optimistic id for the real one.
              set((state) => ({
                messages: state.messages.map((m) =>
                  m.id === optimisticId ? { ...m, id: data.id } : m
                ),
              }))
              break

            case 'trace':
              set((state) => ({
                streaming: state.streaming
                  ? { ...state.streaming, traces: [...state.streaming.traces, data] }
                  : state.streaming,
              }))
              break

            case 'status':
              set((state) => ({
                streaming: state.streaming ? { ...state.streaming, status: data } : state.streaming,
              }))
              break

            case 'token':
              set((state) => ({
                streaming: state.streaming
                  ? { ...state.streaming, text: state.streaming.text + (data.text || '') }
                  : state.streaming,
              }))
              break

            case 'done':
              // The final text has validated, renumbered citation markers, so it
              // replaces what was streamed.
              set((state) => ({
                streaming: state.streaming
                  ? { ...state.streaming, text: data.answer || state.streaming.text }
                  : state.streaming,
              }))
              break

            case 'saved':
              set((state) => ({
                messages: [...state.messages, data.message],
                streaming: null,
                sessions: data.session_title
                  ? state.sessions.map((s) =>
                      s.id === sessionId ? { ...s, title: data.session_title } : s
                    )
                  : state.sessions,
              }))
              break

            case 'error':
              set({ error: data.message })
              break

            default:
              break
          }
        },
        controller.signal,
        forceRoute
      )
    } catch (err) {
      if (err.name !== 'AbortError') {
        set({ error: err.message })
      }
    } finally {
      set({ sending: false, abortController: null, streaming: null })
      // Refresh the ordering so the thread jumps to the top of the sidebar.
      const kbId = get().sessions.find((s) => s.id === sessionId)?.kb_id
      if (kbId) get().loadSessions(kbId)
    }
  },

  stop() {
    const { abortController } = get()
    if (abortController) abortController.abort()
    set({ sending: false, streaming: null, abortController: null })
  },

  clearError() {
    set({ error: null })
  },

  reset() {
    get().stop()
    set({ sessions: [], activeSessionId: null, messages: [], streaming: null, error: null })
  },
}))
