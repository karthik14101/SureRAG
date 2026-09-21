import { api, streamEvents } from './client'

export const chatApi = {
  listSessions: (kbId) =>
    api.get(kbId ? `/chat/sessions?kb_id=${encodeURIComponent(kbId)}` : '/chat/sessions'),
  createSession: (kbId, title) => api.post('/chat/sessions', { kb_id: kbId, title: title || null }),
  renameSession: (id, title) => api.patch(`/chat/sessions/${id}`, { title }),
  deleteSession: (id) => api.del(`/chat/sessions/${id}`),
  messages: (id) => api.get(`/chat/sessions/${id}/messages`),
  citation: (chunkId) => api.get(`/chat/citations/${chunkId}`),

  /** Non-streaming ask, kept for debugging and as a fallback. */
  ask: (sessionId, question, forceRoute) =>
    api.post(`/chat/sessions/${sessionId}/ask`, {
      question,
      force_route: forceRoute || null,
    }),

  /** Streaming ask. `onEvent(name, data)` receives every SSE frame. */
  askStream: (sessionId, question, onEvent, signal, forceRoute) =>
    streamEvents(
      `/chat/sessions/${sessionId}/ask/stream`,
      { question, force_route: forceRoute || null },
      onEvent,
      signal
    ),
}
