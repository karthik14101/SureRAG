import { useEffect, useState } from 'react'
import { fetchBlobUrl } from '../api/client'

/**
 * Load a protected image into an object URL.
 *
 * Images live behind an auth check, so <img src="/api/..."> cannot work: the
 * browser sends no Authorization header on an img request. We fetch the bytes
 * and hand the DOM a blob: URL, revoking it on unmount so memory is not leaked.
 */
export function useAuthedImage(path) {
  const [url, setUrl] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(Boolean(path))

  useEffect(() => {
    if (!path) {
      setUrl(null)
      setLoading(false)
      return undefined
    }

    const controller = new AbortController()
    let objectUrl = null
    setLoading(true)
    setError(null)

    fetchBlobUrl(path, controller.signal)
      .then((blobUrl) => {
        objectUrl = blobUrl
        setUrl(blobUrl)
        setLoading(false)
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          setError(err.message || 'Could not load the image.')
          setLoading(false)
        }
      })

    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [path])

  return { url, loading, error }
}
