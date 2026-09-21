import { api } from './client'

export const kbApi = {
  list: () => api.get('/kb'),
  create: (name, description) => api.post('/kb', { name, description: description || null }),
  get: (id) => api.get(`/kb/${id}`),
  rename: (id, name, description) => api.patch(`/kb/${id}`, { name, description }),
  remove: (id) => api.del(`/kb/${id}`),
  stats: (id) => api.get(`/kb/${id}/stats`),
  rebuildGraph: (id) => api.post(`/kb/${id}/rebuild-graph`),

  documents: (id) => api.get(`/kb/${id}/documents`),
  media: (id) => api.get(`/kb/${id}/media`),

  /** Browse the raw index, or search it when `q` is set. */
  chunks(id, { q, docId, modality, limit = 50, offset = 0 } = {}) {
    const params = new URLSearchParams()
    if (q) params.set('q', q)
    if (docId) params.set('doc_id', docId)
    if (modality) params.set('modality', modality)
    params.set('limit', String(limit))
    params.set('offset', String(offset))
    return api.get(`/kb/${id}/chunks?${params.toString()}`)
  },
  chunkFacets: (id) => api.get(`/kb/${id}/chunks/facets`),
  chunk: (chunkId) => api.get(`/chunks/${chunkId}`),
  deleteDocument: (docId) => api.del(`/documents/${docId}`),
  limits: () => api.get('/config/limits'),

  /** Multipart upload. Returns { job_id, accepted, skipped, total_files }. */
  upload(kbId, files) {
    const form = new FormData()
    for (const file of files) form.append('files', file, file.name)
    return api.post(`/kb/${kbId}/documents`, form)
  },
}
