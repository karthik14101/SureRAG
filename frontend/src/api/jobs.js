import { api } from './client'

export const jobsApi = {
  get: (jobId) => api.get(`/jobs/${jobId}`),
  active: (kbId) => api.get(`/jobs/kb/${kbId}/active`),
}
