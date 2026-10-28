/** Lightweight data-fetching hooks (no external state library needed). */
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'

export interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
  /** status code of the error, if any (0 = network unreachable) */
  errorCode: number | null
  reload: () => void
}

export function useApi<T>(fetcher: () => Promise<T>, enabled = true): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState<string | null>(null)
  const [errorCode, setErrorCode] = useState<number | null>(null)
  const [tick, setTick] = useState(0)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    setErrorCode(null)
    fetcherRef
      .current()
      .then((d) => {
        if (!cancelled) {
          setData(d)
          setLoading(false)
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.message : String(e))
          setErrorCode(e instanceof ApiError ? e.status : null)
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [enabled, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, loading, error, errorCode, reload }
}
