import { api } from './client'

export const healthApi = {
  deep: () => api.get('/health/deep'),
}
