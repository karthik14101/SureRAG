import { api, setToken } from './client'

export const authApi = {
  async signUp({ username, password, displayName }) {
    const data = await api.post('/auth/signup', {
      username,
      password,
      display_name: displayName || null,
    })
    setToken(data.token)
    return data.user
  },

  async signIn({ username, password }) {
    const data = await api.post('/auth/signin', { username, password })
    setToken(data.token)
    return data.user
  },

  async signOut() {
    try {
      await api.post('/auth/signout')
    } finally {
      setToken(null)
    }
  },

  me: () => api.get('/auth/me'),
}
