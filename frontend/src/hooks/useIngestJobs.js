import { useCallback, useEffect, useRef, useState } from 'react'
import { jobsApi } from '../api/jobs'

const TERMINAL = new Set(['completed', 'completed_with_errors', 'failed'])
const POLL_MS = 1200

/**
 * Track ingestion jobs for a knowledge base.
 *
 * Polling rather than SSE here on purpose: progress updates are coarse (one per
 * file), a dropped poll is harmless, and it reconnects for free after a page
 * refresh mid-upload -- which a long-lived stream does not.
 */
export function useIngestJobs(kbId, { onComplete } = {}) {
  const [jobs, setJobs] = useState([])
  const timer = useRef(null)
  const tracked = useRef(new Set())
  const completeRef = useRef(onComplete)
  completeRef.current = onComplete

  const stop = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  const poll = useCallback(async () => {
    const ids = Array.from(tracked.current)
    if (!ids.length) {
      stop()
      return
    }

    try {
      const results = await Promise.all(
        ids.map((id) => jobsApi.get(id).catch(() => null))
      )
      const alive = results.filter(Boolean)
      setJobs(alive)

      let finishedAny = false
      for (const job of alive) {
        if (TERMINAL.has(job.state)) {
          tracked.current.delete(job.id)
          finishedAny = true
        }
      }
      // Jobs that 404'd (deleted KB) must not be polled forever.
      for (let i = 0; i < ids.length; i += 1) {
        if (!results[i]) tracked.current.delete(ids[i])
      }

      if (finishedAny) completeRef.current?.()
    } catch {
      /* transient network error; the next tick retries */
    }

    if (tracked.current.size) {
      timer.current = setTimeout(poll, POLL_MS)
    } else {
      timer.current = null
    }
  }, [stop])

  const track = useCallback(
    (jobId) => {
      if (!jobId) return
      tracked.current.add(jobId)
      if (!timer.current) timer.current = setTimeout(poll, 300)
    },
    [poll]
  )

  // Resume tracking anything already running for this KB (e.g. after a refresh).
  useEffect(() => {
    if (!kbId) return undefined
    let cancelled = false

    jobsApi
      .active(kbId)
      .then((active) => {
        if (cancelled || !active?.length) return
        setJobs(active)
        active.forEach((job) => tracked.current.add(job.id))
        if (!timer.current) timer.current = setTimeout(poll, 300)
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [kbId, poll])

  useEffect(() => stop, [stop])

  const activeJobs = jobs.filter((job) => !TERMINAL.has(job.state))
  const finishedJobs = jobs.filter((job) => TERMINAL.has(job.state))

  return {
    jobs,
    activeJobs,
    finishedJobs,
    isIngesting: activeJobs.length > 0,
    track,
    dismiss: (jobId) => setJobs((prev) => prev.filter((job) => job.id !== jobId)),
    clearFinished: () => setJobs((prev) => prev.filter((job) => !TERMINAL.has(job.state))),
  }
}
