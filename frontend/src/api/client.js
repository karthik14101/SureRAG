/**
 * HTTP client.
 *
 * Deliberately plain fetch rather than axios: the streaming endpoint needs a
 * ReadableStream reader, which axios does not expose in the browser, and there
 * is no other reason to carry the dependency.
 */

const BASE = '/api/v1'
const TOKEN_KEY = 'sure_token'

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    // Private browsing can throw on localStorage access.
    return null
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable; the session simply will not survive a reload */
  }
}

export class ApiError extends Error {
  constructor(message, status, code, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail || {}
  }
}

function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

/** Turn a non-OK response into a typed ApiError carrying the server's message. */
async function toError(response) {
  let payload = null
  try {
    payload = await response.json()
  } catch {
    /* non-JSON error body */
  }
  const envelope = payload?.error
  const message =
    envelope?.message ||
    payload?.detail ||
    `Request failed (${response.status} ${response.statusText})`

  // A dead session should bounce the user to sign-in immediately.
  if (response.status === 401) {
    setToken(null)
    window.dispatchEvent(new CustomEvent('sure:unauthorized'))
  }

  return new ApiError(message, response.status, envelope?.code, envelope?.detail)
}

async function request(path, { method = 'GET', body, headers, signal } = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: authHeaders({
        ...(body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...headers,
      }),
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new ApiError(
      'Cannot reach the backend. Is it running on http://127.0.0.1:8000?',
      0,
      'network_error'
    )
  }

  if (!response.ok) throw await toError(response)
  if (response.status === 204) return null

  const contentType = response.headers.get('content-type') || ''
  return contentType.includes('application/json') ? response.json() : response.text()
}

export const api = {
  get: (path, options) => request(path, { ...options, method: 'GET' }),
  post: (path, body, options) => request(path, { ...options, method: 'POST', body }),
  patch: (path, body, options) => request(path, { ...options, method: 'PATCH', body }),
  del: (path, options) => request(path, { ...options, method: 'DELETE' }),
}

/**
 * Fetch a protected binary resource (an image) as an object URL.
 * EventSource and <img src> cannot send an Authorization header, so images are
 * fetched here and handed to the DOM as blob: URLs.
 */
export async function fetchBlobUrl(path, signal) {
  const response = await fetch(`${BASE}${path}`, { headers: authHeaders(), signal })
  if (!response.ok) throw await toError(response)
  const blob = await response.blob()
  return URL.createObjectURL(blob)
}

/**
 * POST and consume a Server-Sent Event stream.
 *
 * EventSource only does GET and cannot set headers, so the SSE frames are
 * parsed by hand off the response body.
 *
 * @param {string} path
 * @param {object} body
 * @param {(event: string, data: any) => void} onEvent
 * @param {AbortSignal} signal
 */
export async function streamEvents(path, body, onEvent, signal) {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) throw await toError(response)
  if (!response.body) throw new ApiError('Streaming is not supported by this browser.', 0)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Frames are separated by a blank line.
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)

      let eventName = 'message'
      const dataLines = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue

      try {
        onEvent(eventName, JSON.parse(dataLines.join('\n')))
      } catch {
        onEvent(eventName, dataLines.join('\n'))
      }
    }
  }
}

/** GET-based SSE (used for ingestion progress), returns a cleanup function. */
export function subscribeEvents(path, onEvent, onError) {
  const controller = new AbortController()

  ;(async () => {
    try {
      const response = await fetch(`${BASE}${path}`, {
        headers: authHeaders({ Accept: 'text/event-stream' }),
        signal: controller.signal,
      })
      if (!response.ok || !response.body) throw await toError(response)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        let boundary
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)

          let eventName = 'message'
          const dataLines = []
          for (const line of frame.split('\n')) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim()
            else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
          }
          if (!dataLines.length) continue
          try {
            onEvent(eventName, JSON.parse(dataLines.join('\n')))
          } catch {
            onEvent(eventName, dataLines.join('\n'))
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') onError?.(err)
    }
  })()

  return () => controller.abort()
}
