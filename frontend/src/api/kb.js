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

  /* Knowledge-graph explorer. Entities are keyed by normalised name. */
  graphFacets: (id) => api.get(`/kb/${id}/graph/facets`),
  graphEntities(id, { q, type, sort = 'degree', limit = 30, offset = 0 } = {}) {
    const params = new URLSearchParams({ sort, limit: String(limit), offset: String(offset) })
    if (q) params.set('q', q)
    if (type) params.set('type', type)
    return api.get(`/kb/${id}/graph/entities?${params.toString()}`)
  },
  graphEntity: (id, entityId) =>
    api.get(`/kb/${id}/graph/entity?${new URLSearchParams({ id: entityId }).toString()}`),
  graphRelations(id, { q, predicate, limit = 50, offset = 0 } = {}) {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
    if (q) params.set('q', q)
    if (predicate) params.set('predicate', predicate)
    return api.get(`/kb/${id}/graph/relations?${params.toString()}`)
  },
  graphOverview: (id, limit = 40) => api.get(`/kb/${id}/graph/overview?limit=${limit}`),
  graphNeighbourhood(id, entityId, { hops = 1, limit = 40 } = {}) {
    const params = new URLSearchParams({ id: entityId, hops: String(hops), limit: String(limit) })
    return api.get(`/kb/${id}/graph/neighbourhood?${params.toString()}`)
  },
  deleteDocument: (docId) => api.del(`/documents/${docId}`),
  limits: () => api.get('/config/limits'),

  /** Multipart upload. Returns { job_id, accepted, skipped, total_files }. */
  upload(kbId, files) {
    const form = new FormData()
    for (const file of files) form.append('files', file, file.name)
    return api.post(`/kb/${kbId}/documents`, form)
  },
}
