import { useEffect, useState } from 'react'

/**
 * Fetch a JSON summary produced by the analysis pipeline
 * (served from webpage/public/data/).
 */
export default function useJson(file) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetch(`/data/${file}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((json) => {
        if (!cancelled) setData(json)
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [file])

  return { data, loading }
}